from typing import List, Optional, Tuple, Union

import torch 
import torch.nn as nn 
import numpy as np

from utils import randn_tensor

from.scheduling_ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler):    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_inference_steps is not None, "Please set `num_inference_steps` before running inference using DDIM."
        self.set_timesteps(self.num_inference_steps)

    
    def _get_variance(self, t):
        """
        Compute variance for DDIM scheduler at timestep t.
        """
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        
        # DDIM variance computation
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        eta: float=0.0,
    ) -> torch.Tensor:
        """
        DDIM step computation.
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        beta_prod_t = 1 - alpha_prod_t
        
        # 2. compute predicted original sample from predicted noise
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
            pred_epsilon = model_output
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # 3. Clip predicted x0
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        # 4. compute variance: "sigma_t(η)" 
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(t)
        std_dev_t = eta * torch.sqrt(variance)

        # 5. compute "direction pointing to x_t"
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon

        # 6. compute x_t without "random noise"
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        # 7. Add noise
        if eta > 0:
            noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype
            )
            variance = std_dev_t * noise
            prev_sample = prev_sample + variance

        return prev_sample