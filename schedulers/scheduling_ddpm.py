from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from utils import randn_tensor


class DDPMScheduler(nn.Module):

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        """
        Args:
            num_train_timesteps (`int`):

        """
        super(DDPMScheduler, self).__init__()

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.beta = beta_start
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # TODO: calculate betas
        if self.beta_schedule == "linear":
            # This is the DDPM implementation
            betas = torch.linspace(
                self.beta_start, self.beta_end, self.num_train_timesteps
            )
        self.register_buffer("betas", betas)

        # TODO: calculate alphas
        alphas = 1 - betas
        self.register_buffer("alphas", alphas)
        # TODO: calculate alpha cumulative product
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # TODO: timesteps
        self.register_buffer("timesteps", torch.arange(self.num_train_timesteps).long())

    def set_timesteps(
        self,
        num_inference_steps: int = 250,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        # TODO: set timesteps
        timesteps = torch.linspace(
            0, self.num_train_timesteps - 1, num_inference_steps
        ).long()
        self.timesteps = timesteps.to(device)
        # timesteps = None
        # if device:
        #     self.timesteps = torch.from_numpy(timesteps).to(device)
        # else:
        #     self.timesteps = torch.from_numpy(timesteps)

    def __len__(self):
        return self.num_train_timesteps

    def previous_timestep(self, timestep):
        """
        Get the previous timestep for a given timestep.

        Args:
            timestep (`int`): The current timestep.

        Return:
            prev_t (`int`): The previous timestep.
        """
        num_inference_steps = (
            self.num_inference_steps
            if self.num_inference_steps
            else self.num_train_timesteps
        )
        # TODO: caluclate previous timestep
        if timestep == 0:
            prev_t = 0
        else:
            prev_t = timestep - 1
        return prev_t

    def _get_variance(self, t):
        """
        Calculate variance for timestep t according to DDPM paper.
        """
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Calculate variance based on variance_type
        if self.variance_type == "fixed_small":
            # For fixed_small, variance is simply beta_t
            variance = current_beta_t
        elif self.variance_type == "fixed_large":
            # For fixed_large, use the formula from the DDPM paper
            variance = current_beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            # Small hack for better decoder log likelihood
            if t == 1:
                variance = current_beta_t
        else:
            raise NotImplementedError(f"Variance type {self.variance_type} not implemented.")

        return variance

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples based on the given timesteps.

        Args:
            original_samples (torch.Tensor): Original images. Shape: (batch_size, C, H, W)
            noise (torch.Tensor): Noise tensor. Shape: (batch_size, C, H, W)
            timesteps (torch.IntTensor): Timesteps for each sample in the batch. Shape: (batch_size,)

        Returns:
            torch.Tensor: Noisy images.
        """
        # Ensure alphas_cumprod is on the same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(
            dtype=original_samples.dtype, device=original_samples.device
        )

        # Select alphas_cumprod for each timestep in the batch
        # Shape after indexing: (batch_size,)
        alphas_cumprod_t = alphas_cumprod[timesteps]

        # Compute sqrt(alpha_t) and sqrt(1 - alpha_t)
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod_t)  # Shape: (batch_size,)
        sqrt_one_minus_alpha_prod = torch.sqrt(
            1 - alphas_cumprod_t
        )  # Shape: (batch_size,)

        # Reshape for broadcasting: (batch_size, 1, 1, 1)
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)

        # Add noise to the original samples
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )

        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = self.alphas[t]
        current_beta_t = self.betas[t]

        # 2. compute predicted original sample from predicted noise
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # 3. Clip predicted x0
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev) * current_beta_t / (1 - alpha_prod_t)
        current_sample_coeff = torch.sqrt(alpha_prod_t) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)

        # 5. Compute predicted previous sample µ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance = self._get_variance(t) ** 0.5 * noise

        # Add variance to pred_prev_sample
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
