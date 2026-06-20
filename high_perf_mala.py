import os
import gc
import functools
import math
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent GUI window popup on headless environment
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch

from norm import renorm
from sde import marginal_prob_std, diffusion_coeff
from unet import ScoreNet
from observable import jackknife_stats
from mc import gauge_cooling

# Setup style with error handling in case fonts/styles are missing
try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'grid'])
except Exception as e:
    print(f"Warning: Could not set scienceplots style: {e}. Falling back to default style.")
    plt.style.use('ggplot')

# Set DPI parameters
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# --- Physics Parameters ---
L = 16
model_beta = 1
model_L = 16
target_beta = 7
sigma = 25.0

# 1. Device Setup (Enforce MPS if available)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)

# Load pretrained ScoreNet model weights
print("Loading model weights...")
ckpt_path = 'models/noise_ckpt_L16_beta1_num153600_nm1_epoch249.pth'
ckpt_1 = torch.load(ckpt_path, map_location=device, weights_only=True)

# Clean keys (remove "module." prefix if present because we aren't using DataParallel on MPS)
new_state_dict = OrderedDict()
for k, v in ckpt_1.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
score_model.load_state_dict(new_state_dict)
score_model = score_model.to(device)
score_model.eval()

# --- Ultimate Optimized MALA Sampler ---
def action(phi, beta):
    return beta * (-torch.sum(torch.cos(torch.pi *(phi[:,0, :, :] - phi[:,1, :, :] +
                                         torch.roll(phi[:,1, :, :], shifts=-1, dims=1) -
                                           torch.roll(phi[:,0, :, :], shifts=-1, dims=2))), dim=(1, 2)))

def ultimate_MALA_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size,
                           num_steps, beta,
                           ratio, alpha, mh_steps, t_mh,
                           L, start, size,
                           device='cuda'):
    indices = torch.arange(num_steps, device=device).float()
    time_steps = torch.pow(ratio, indices) * start
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 2, L, L, device=device) * marginal_prob_std(t)[:, None, None, None]
    
    x = init_x
    score_x = None
    S_x = None
    
    # Track acceptance using tensors on GPU to avoid CPU-GPU synchronization inside loops
    total_accepted = torch.zeros(1, device=device)
    total_mh_steps = 0
    
    rate_traj = []
    s_list = []
    Q_list = []
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="Time Steps"):
            step_size = alpha * (time_step / time_steps[-1]) ** 2
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            
            if time_step < t_mh:
                current_size = mh_steps
            else:
                current_size = size
                
            # Pre-evaluate score_x at the start of the time step
            score_x = score_model(x, batch_time_step)
            if time_step < t_mh:
                S_x = action(x, beta)
            
            sqrt_2_step_size = math.sqrt(2.0 * step_size)
            
            for turn in range(current_size):
                mean_x = x + beta * score_x * step_size
                randn_noise = torch.randn_like(x)
                x_temp = mean_x + sqrt_2_step_size * randn_noise
                
                if time_step < t_mh:
                    score_y = score_model(x_temp, batch_time_step)
                    drift_y = beta * step_size * (score_x + score_y)
                    p_1 = -(1.0 / step_size) * 0.25 * torch.sum((sqrt_2_step_size * randn_noise + drift_y)**2, dim=(1, 2, 3))
                    p_2 = 0.5 * torch.sum(randn_noise ** 2, dim=(1, 2, 3))
                    delta_q = torch.exp(p_1 + p_2)
                    
                    # S_x is read from cache, only S_y is evaluated
                    S_y = action(x_temp, beta)
                    delta_pi = torch.exp(S_x - S_y)
                    
                    accept_prob = delta_pi * delta_q
                    random_i = torch.rand(batch_size, device=device)
                    
                    accept_mask = random_i < accept_prob
                    
                    # Use torch.where to keep operations fully vectorized on GPU (avoids slow boolean indexing)
                    mask_4d = accept_mask[:, None, None, None]
                    x = torch.where(mask_4d, x_temp, x)
                    score_x = torch.where(mask_4d, score_y, score_x)
                    S_x = torch.where(accept_mask, S_y, S_x)
                    
                    # Accumulate acceptance on GPU
                    total_accepted += accept_mask.sum()
                    total_mh_steps += 1
                else:
                    x = x_temp
                    if turn < current_size - 1:
                        score_x = score_model(x, batch_time_step)
            
            x_cpu = (x + beta * score_x * step_size).detach().cpu().numpy()
            
    # Compute acceptance rate on CPU only at the very end
    final_rate = total_accepted.item() / (batch_size * max(1, total_mh_steps))
    return x_cpu, final_rate, rate_traj, s_list, Q_list


# --- Fine-grained Batch Configuration ---
total_samples = 1024
batch_size = 256  # Memory safe and optimal speed for 8GB M2 Mac
num_batches = total_samples // batch_size
save_dir = 'data/chunks'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

alpha = 5e-9
ratio = 0.9
t_0 = 1
size = 400
num_steps = 100

print(f"Total batches to run: {num_batches} (Batch size: {batch_size})")

# 2. MALA Sampling Loop with strict memory management
with torch.inference_mode():
    for i in range(num_batches):
        print(f"\n--- Running batch {i+1}/{num_batches} ---")
        
        # Run ultimate optimized sampler for this batch
        samples, rate, rate_traj, s_list, Q_list = ultimate_MALA_sampler(
            score_model, 
            marginal_prob_std_fn,
            diffusion_coeff_fn, 
            batch_size=batch_size, 
            num_steps=num_steps, 
            device=device,
            alpha=alpha, ratio=ratio, mh_steps=400, t_mh=0.001,
            L=L, start=t_0, size=size, beta=target_beta/model_beta
        )
        
        # Check if samples is a PyTorch tensor or a NumPy array
        if hasattr(samples, 'cpu'):
            batch_np = samples.cpu().numpy()
        else:
            batch_np = np.array(samples)
            
        # Ensure correct shape (batch_size, 2, L, L)
        batch_np = batch_np.reshape(batch_size, 2, L, L)
        
        # Save current chunk to disk immediately
        chunk_file = os.path.join(save_dir, f'cfg_chunk_{i}.npy')
        np.save(chunk_file, batch_np)
        print(f"Batch {i+1} saved to {chunk_file}. Acceptance rate: {rate:.4f}")
        
        # Free resources
        del samples, rate_traj, s_list, Q_list
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

print("\nAll batches finished. Merging files...")

# 3. Merge chunk files to get the final generated configs
all_samples = []
for i in range(num_batches):
    chunk_file = os.path.join(save_dir, f'cfg_chunk_{i}.npy')
    chunk_data = np.load(chunk_file)
    all_samples.append(chunk_data)

final_cfgs = np.concatenate(all_samples, axis=0)
output_path = 'data/cfg_L16_beta1_1k_final.npy'
np.save(output_path, final_cfgs)
print(f"Sampling Done! Final shape of DM configs: {final_cfgs.shape}")

# Cleanup chunk files to keep workspace tidy
for i in range(num_batches):
    chunk_file = os.path.join(save_dir, f'cfg_chunk_{i}.npy')
    if os.path.exists(chunk_file):
        os.remove(chunk_file)
if os.path.exists(save_dir):
    try:
        os.rmdir(save_dir)
    except Exception:
        pass

# --- 4. Physics Observables Analysis ---
print("\nAnalyzing physics observables...")

# Load pre-trained HMC baseline configurations
cfgs_baseline_path = 'data/cfgs_L16_beta1_1k.npy'
if os.path.exists(cfgs_baseline_path):
    print(f"Loading baseline from {cfgs_baseline_path}...")
    cfgs = np.load(cfgs_baseline_path)
    cfgs = gauge_cooling(cfgs[:1024])
else:
    raise FileNotFoundError(f"Baseline configs file {cfgs_baseline_path} not found!")

# Process generated (DM) configurations
cfgs_min = -np.pi
cfgs_max = np.pi

samples_re = renorm(final_cfgs, cfgs_min, cfgs_max)
cfgs_test = np.transpose(samples_re, (0, 2, 3, 1))

# Definition of physics helpers for calculations
def wilson_loop(lattice, x, y):
    U1 = np.exp(1j * lattice[x, y, 0])
    U2 = np.exp(1j * lattice[x, (y + 1) % L, 0])
    U3 = np.exp(1j * lattice[(x + 1) % L, y, 1])
    U4 = np.exp(1j * lattice[x, y, 1])
    return np.real(U1 * U3 * np.conj(U2) * np.conj(U4))

def calculate_angle(array1, xp, yp):
    right = array1[xp % L, yp % L][0]
    up = array1[(xp + 1) % L, yp % L][1]
    down = array1[xp % L, yp % L][1]
    left = array1[xp % L, (yp + 1) % L][0]
    return right + up - left - down

def project_angle(angle):
    return np.remainder(angle + np.pi, (2 * np.pi)) - np.pi

def calculate_topological_charge(array):
    top_charge = 0
    for i in range(L):
        for j in range(L):
            angle = calculate_angle(array, i, j)
            top_charge += project_angle(angle)
    return round(top_charge / (2 * np.pi))

def topological_suspectiblity(array):
    top_charge = 0
    for i in range(L):
        for j in range(L):
            angle = calculate_angle(array, i, j)
            top_charge += project_angle(angle)
    return round(top_charge / (2 * np.pi))**2

def wilson_loopij(array, x_len, t_len):
    Wilson_loop_values = []
    for conf in range(array.shape[0]):
        init = np.zeros((L, L))
        for x in range(x_len):
            init += np.roll(array[conf, :, :, 0], -x, axis=0)
        for t in range(t_len):
            init += np.roll(np.roll(array[conf, :, :, 1], -x_len, axis=0), -t, axis=1)
        for x in range(x_len):
            init -= np.roll(np.roll(array[conf, :, :, 0], (-x_len, -t_len), axis=(0, 1)), x + 1, axis=0)   
        for t in range(t_len):
            init -= np.roll(np.roll(array[conf, :, :, 1], -t_len, axis=1), t + 1, axis=1)
        wil_l = np.mean(np.cos(init), axis=(0, 1))
        Wilson_loop_values.append(wil_l)
    return Wilson_loop_values

print("Computing Wilson Loops...")
cfgs_plaq = wilson_loopij(cfgs, 1, 1)
cfgs_test_plaq = wilson_loopij(cfgs_test, 1, 1)

# Print comparison of Wilson Loops
print("\n--- Wilson Loop average & standard deviation comparison ---")
print(f"{'Loop Size':<10} | {'DM model (mean, std)':<30} | {'HMC Baseline (mean, std)':<30}")
print("-" * 79)
for i in range(1, 5):
    dm_mean = np.mean(wilson_loopij(cfgs_test, i, i))
    dm_std = np.std(wilson_loopij(cfgs_test, i, i))
    hmc_mean = np.mean(wilson_loopij(cfgs, i, i))
    hmc_std = np.std(wilson_loopij(cfgs, i, i))
    print(f"{i}x{i:<8} | {dm_mean:.6f}, {dm_std:.6f} | {hmc_mean:.6f}, {hmc_std:.6f}")

# Topological charge calculations
print("\nCalculating Topological Charges...")
top_charge_history = []
top_charge_history2 = []
for conf in range(cfgs_test.shape[0]):
    top_charge_history.append(calculate_topological_charge(cfgs_test[conf]))
for conf in range(cfgs.shape[0]):
    top_charge_history2.append(calculate_topological_charge(cfgs[conf]))

# Topological susceptibility calculations
print("\nCalculating Topological Susceptibilities (Jackknife)...")
suspect1 = []
for conf in range(cfgs_test.shape[0]):
    suspect1.append(topological_suspectiblity(cfgs_test[conf]))
mean1, err1 = jackknife_stats(suspect1)
print(f"DM Topological Susceptibility / L^2: {mean1 / L**2:.6f} +/- {err1 / L**2:.6f}")

suspect2 = []
for conf in range(cfgs.shape[0]):
    suspect2.append(topological_suspectiblity(cfgs[conf]))
mean2, err2 = jackknife_stats(suspect2)
print(f"HMC Topological Susceptibility / L^2: {mean2 / L**2:.6f} +/- {err2 / L**2:.6f}")

# --- 5. Generate and Save Figures ---
print("\nGenerating figures...")

# Plot 1: Wilson Loop histogram
plt.figure(figsize=(8, 6))
Xrange = (0.85, 1)
plt.hist(cfgs_plaq, bins=40, range=Xrange, histtype='step', linewidth=2, edgecolor='blue', color='blue', alpha=1, density=True, label=r'HMC $\beta = 7$')
plt.hist(cfgs_test_plaq, bins=40, range=Xrange, edgecolor='black', color='orange', alpha=1, density=True, label=r'DM $\beta = 7$')
ax = plt.gca()  
ax.yaxis.set_major_locator(MultipleLocator(5)) 
ax.yaxis.set_minor_locator(MultipleLocator(2.5)) 
plt.legend()
plt.xlabel(r'$\langle W_{1 \times 1} \rangle$', fontsize=16)
plt.ylabel(r'Frequency density', fontsize=16)
plt.savefig('beta_7_Wilson_loop_step.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()
print("Saved beta_7_Wilson_loop_step.pdf")

# Plot 2: Topological Charge histogram
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
bin_range = range(-5, 6)
hist2, bins2 = np.histogram(top_charge_history2, bins=bin_range, density=True)
hist, bins = np.histogram(top_charge_history, bins=bin_range, density=True)

ax.bar(bins2[:-1] - 0.2, hist2, width=0.5, edgecolor='black', color='blue', alpha=1, label=r'HMC $\beta=7$')
ax.bar(bins[:-1] + 0.2, hist, width=0.5, edgecolor='black', color='orange', alpha=1, label=r'DM $\beta=7$')

ax.set_xlabel(r'Topological Charge $Q$', fontsize=16)
ax.set_ylabel(r'Frequency density', fontsize=16)
ax.set_xticks(range(-5, 6, 1))
ax.set_xlim(-5.5, 5.5)
ax.yaxis.set_major_locator(MultipleLocator(0.2)) 
ax.yaxis.set_minor_locator(MultipleLocator(0.05))  
plt.legend()
plt.savefig('beta_7_Q_new.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()
print("Saved beta_7_Q_new.pdf")

print("All calculations and figures finished successfully!")
