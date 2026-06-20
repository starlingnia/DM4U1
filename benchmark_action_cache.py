import time
import functools
import torch
import numpy as np
from sde import marginal_prob_std, diffusion_coeff
from unet import ScoreNet

def action(phi, beta):
    return beta * (-torch.sum(torch.cos(torch.pi *(phi[:,0, :, :] - phi[:,1, :, :] +
                                         torch.roll(phi[:,1, :, :], shifts=-1, dims=1) -
                                           torch.roll(phi[:,0, :, :], shifts=-1, dims=2))), dim=(1, 2)))

def sampler_with_action_cache(score_model,
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
    
    rate = 0
    rate_traj = []
    s_list = []
    Q_list = []
    x = init_x
    score_x = None
    S_x = None
    
    with torch.no_grad():
        for time_step in time_steps:
            MH_step = 1e-9
            rate = 0
            step_size = alpha * (time_step / time_steps[-1]) ** 2
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            
            if time_step < t_mh:
                current_size = mh_steps
            else:
                current_size = size
                
            score_x = score_model(x, batch_time_step)
            # Initialize S_x cache if we are going into MH phase
            if time_step < t_mh:
                S_x = action(x, beta)
            
            for turn in range(current_size):
                mean_x = x + beta * score_x * step_size
                randn_noise = torch.randn_like(x)
                x_temp = mean_x + torch.sqrt(torch.tensor(2 * step_size, device=device)) * randn_noise
                
                if time_step < t_mh:
                    score_y = score_model(x_temp, batch_time_step)
                    drift_y = beta * step_size * (score_x + score_y)
                    p_1 = -(1.0 / step_size) * 0.25 * torch.sum((torch.sqrt(torch.tensor(2 * step_size, device=device)) * randn_noise + drift_y)**2, dim=(1, 2, 3))
                    p_2 = 0.5 * torch.sum(randn_noise ** 2, dim=(1, 2, 3))
                    delta_q = torch.exp(p_1 + p_2)
                    
                    # S_x is read from cache, only S_y is evaluated
                    S_y = action(x_temp, beta)
                    delta_pi = torch.exp(S_x - S_y)
                    
                    accept_prob = delta_pi * delta_q
                    random_i = torch.rand(batch_size, device=device)
                    
                    accept_mask = random_i < accept_prob
                    x = x.clone()
                    x[accept_mask] = x_temp[accept_mask]
                    
                    score_x = score_x.clone()
                    score_x[accept_mask] = score_y[accept_mask]
                    
                    # Update S_x cache
                    S_x = S_x.clone()
                    S_x[accept_mask] = S_y[accept_mask]
                    
                    acc_rate = accept_mask.sum().item() / batch_size
                    MH_step += 1
                    rate += acc_rate
                else:
                    x = x_temp
                    if turn < current_size - 1:
                        score_x = score_model(x, batch_time_step)
            
            x_cpu = (x + beta * score_x * step_size).detach().cpu().numpy()
            
    return x_cpu, rate / MH_step, rate_traj, s_list, Q_list

def run_test_cache(device_name, batch_size):
    device = torch.device(device_name)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25.0, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=25.0, device=device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    score_model.eval()
    
    alpha = 5e-9
    ratio = 0.9
    t_0 = 1.0
    size = 40
    num_steps = 1
    
    # Warmup
    with torch.no_grad():
        score_model(torch.randn(batch_size, 2, 16, 16, device=device), torch.ones(batch_size, device=device))
        
    start_time = time.time()
    samples, rate, rate_traj, s_list, Q_list = sampler_with_action_cache(
        score_model, 
        marginal_prob_std_fn,
        diffusion_coeff_fn, 
        batch_size=batch_size, 
        num_steps=num_steps, 
        device=device,
        alpha=alpha, ratio=ratio, mh_steps=size, t_mh=2.0,  # Force MH
        L=16, start=t_0, size=size, beta=7.0
    )
    elapsed = time.time() - start_time
    print(f"Action Cached | Batch Size: {batch_size} on {device_name} | 1 step (size={size}) took: {elapsed:.4f} seconds")

print("Running action-cached benchmark on MPS...")
run_test_cache("mps", 128)
