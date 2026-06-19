import unittest
import numpy as np
import torch
import torch.nn as nn
import functools

# Import project modules
from norm import norm, renorm
from sde import marginal_prob_std, diffusion_coeff
from unet import ScoreNet
from loss import loss_fn
from mc import get_action, gauge_cooling, get_drift, hmc
from observable import jackknife, jackknife_stats, compute_autocorrelation, compute_ess
from sampler import calculate_angle, calculate_topological_charge

class TestLatticeGaugeDiffusion(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.L = 8  # Use a smaller size for quick testing
        self.beta = 1.0
        self.sigma = 25.0

    def test_norm_and_renorm(self):
        """Test normalization and renormalization mapping."""
        x = np.array([-np.pi, 0.0, np.pi])
        y, xmin, xmax = norm(x)
        self.assertAlmostEqual(y[0], -1.0)
        self.assertAlmostEqual(y[1], 0.0)
        self.assertAlmostEqual(y[2], 1.0)
        
        # Test reconstruction
        x_recon = renorm(y, xmin, xmax)
        np.testing.assert_allclose(x, x_recon)

    def test_sde_functions(self):
        """Test SDE marginal probability and diffusion coefficients."""
        t_val = 0.5
        std = marginal_prob_std(t_val, self.sigma, self.device)
        coeff = diffusion_coeff(t_val, self.sigma, self.device)
        
        self.assertTrue(torch.is_tensor(std))
        self.assertTrue(torch.is_tensor(coeff))
        self.assertEqual(std.device.type, self.device)
        self.assertEqual(coeff.device.type, self.device)
        self.assertTrue(std > 0)
        self.assertTrue(coeff > 0)

    def test_unet_scorenet(self):
        """Test ScoreNet network structure and forward pass."""
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=self.sigma, device=self.device)
        model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
        model = model.to(self.device)
        
        batch_size = 4
        x = torch.randn(batch_size, 2, self.L, self.L, device=self.device)
        t = torch.rand(batch_size, device=self.device)
        
        output = model(x, t)
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_loss_function(self):
        """Test loss calculation and gradients."""
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=self.sigma, device=self.device)
        model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
        model = model.to(self.device)
        
        batch_size = 4
        x = torch.randn(batch_size, 2, self.L, self.L, device=self.device)
        
        loss = loss_fn(model, x, marginal_prob_std_fn)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)  # Should be scalar
        self.assertTrue(loss.item() >= 0)
        
        # Test backward pass
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

    def test_monte_carlo(self):
        """Test basic MC operations: action, drift, cooling, and hmc step."""
        phi = np.random.randn(self.L, self.L, 2) * np.pi
        
        # Action and drift
        S = get_action(phi, self.beta)
        drift = get_drift(phi, self.beta)
        self.assertTrue(np.isscalar(S))
        self.assertEqual(drift.shape, phi.shape)
        
        # Cooling
        cooled = gauge_cooling(phi)
        self.assertTrue((cooled >= -np.pi).all() and (cooled <= np.pi).all())
        
        # HMC step
        new_phi, new_S, accepted = hmc(phi, S, self.beta, n_steps=2)
        self.assertEqual(new_phi.shape, phi.shape)
        self.assertTrue(np.isscalar(new_S))
        self.assertTrue(isinstance(accepted, bool))

    def test_observables(self):
        """Test jackknife stats and autocorrelation/ESS calculations."""
        data = np.random.randn(100)
        
        # Jackknife
        mean, error = jackknife_stats(data)
        self.assertAlmostEqual(mean, np.mean(data), places=5)
        self.assertTrue(error > 0)
        
        # Autocorrelation and ESS
        autocorr = compute_autocorrelation(data, max_lag=10)
        self.assertEqual(len(autocorr), 11)
        self.assertAlmostEqual(autocorr[0], 1.0)
        
        ess, tau_int = compute_ess(data, max_lag=10)
        self.assertTrue(ess > 0)
        self.assertTrue(tau_int > 0)

    def test_sampler_angle_and_topological_charge(self):
        """Test angle calculation and topological charge logic."""
        batch_size = 2
        phi = torch.randn(batch_size, 2, self.L, self.L, device=self.device)
        
        angles = calculate_angle(phi)
        self.assertEqual(angles.shape, (batch_size, self.L, self.L))
        
        q = calculate_topological_charge(phi)
        self.assertEqual(q.shape, (batch_size,))
        self.assertFalse(torch.isnan(q).any())

if __name__ == '__main__':
    unittest.main()
