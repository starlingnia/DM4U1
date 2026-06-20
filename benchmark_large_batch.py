import time
import functools
import torch
import numpy as np
from sde import marginal_prob_std, diffusion_coeff
from unet import ScoreNet
from benchmark_action_cache import sampler_with_action_cache

def run_test_large(device_name, batch_size):
    device = torch.device(device_name)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25.0, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=25.0, device=device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    score_model.eval()
    
    alpha = 5e-9
    ratio = 0.9
    t_0 = 1.0
    size = 10  # use smaller size for quick check
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
    print(f"Large Batch Size: {batch_size} on {device_name} | 1 step (size={size}) took: {elapsed:.4f} seconds")

print("Running large batch tests on MPS...")
for bs in [512, 1024]:
    try:
        run_test_large("mps", bs)
    except Exception as e:
        print(f"Failed for batch size {bs}: {e}")
