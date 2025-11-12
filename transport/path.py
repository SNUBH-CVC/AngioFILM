
# Transport path implementations for flow matching and diffusion models.
# 
# This file contains code adapted from the SiT (Scalable Interpolant Transformers) project:
# Original source: https://github.com/willisma/SiT/blob/main/transport/path.py
# License: MIT (as per the original repository)
# 
# Some classes and methods have been modified for this project's specific requirements.

from functools import partial

import numpy as np
import torch as th


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


#################### Coupling Plans ####################

class ICPlan:
    """Linear Coupling Plan"""
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var  # (55)
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift  # (54)
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0
    
    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt
    
    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0  # (47)
    
    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut
    

class VPCPlan(ICPlan):
    """class for VP path flow matching"""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min 
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min


    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = self.log_mean_coeff(t)
        alpha_t = th.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = th.sqrt(1 - th.exp(p_sigma_t))
        d_sigma_t = th.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2
    

class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
    
    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = th.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        sigma_t = th.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * th.tan(t * np.pi / 2))


class BridgePlan:
    """
    Brownian-bridge coupling plan
    """
    def __init__(self, sigma=0.05):
        self.sigma = sigma  # (2a)**0.5

    def plan(self, t, x0, x1):
        """
        Training-time forward process at arbitrary t in [0, 1]:
          mean = (1-t) * x0 + t * x1
          var  = σ² * t * (1-t)
          x_t  = mean + sqrt(var) * noise
        Return (t, x_t, x1 - x_t).
        """
        t_b = expand_t_like_x(t, x0)
        mean = (1 - t_b) * x0 + t_b * x1
        var = (self.sigma ** 2) * t_b * (1 - t_b)
        noise = th.randn_like(x0)
        xt = mean + th.sqrt(var) * noise
        ut_wo_denom = x1 - xt
        return t, xt, ut_wo_denom

    def sample(self, model, x0, num_steps=50, eps=1e-3):
        """
        Sample from the bridge by stepping t=0→1 in num_steps increments.
        model(xt, t) must predict u = x1 - xt.

        Args:
            model: Model function that predicts u = x1 - xt
            x0: Initial point
            num_steps: Number of sampling steps
            eps: Small epsilon for numerical stability (defaults to 1e-6)

        W(s) = (s_{i+1} - s) * x_i + (s - s_i) * x_{i+1} / (s_{i+1} - s_i) + sqrt((s_{i+1} - s) * (s - s_i)) / (s_{i+1} - s_i)) * z_i
        You can compute the mean and variance accordingly.
        """
        # Input validation
        if num_steps == 1:
            # Special case: single step, just predict x1 directly
            with th.no_grad():
                u = model(x0, th.tensor(0.0, device=x0.device))
                return x0 + u
        
        device = x0.device
        ts = th.linspace(0.0, 1.0, num_steps + 1, device=device)
        xt = x0.clone()

        for i in range(num_steps):
            t_curr = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_curr

            # 1) predict drift toward endpoint
            u = model(xt, t_curr)   # should return x1 - xt
            x1_hat = xt + u           # estimate of x1

            # if this is the final step, just return x1_hat
            if i == num_steps - 1:
                return x1_hat

            # 2) compute bridge‐step mean & variance
            # E[x_{t_next}|x_t, x1] = x_t + ((t_next - t_curr)/(1 - t_curr)) * (x1_hat - x_t)
            # Var = σ² * (1 - t_next)*(t_next - t_curr)/(1 - t_curr)
            one_minus_t_curr = 1.0 - t_curr
            
            # Add small epsilon to prevent division by very small numbers
            if one_minus_t_curr < eps:
                # If we're very close to t=1, just return the prediction
                return x1_hat
            
            mean = xt + (dt / one_minus_t_curr) * u  
            var = (self.sigma ** 2) * (1.0 - t_next) * dt / one_minus_t_curr

            noise = th.randn_like(xt)
            xt = mean + noise * var.sqrt()
            
        return xt