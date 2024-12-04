from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import randn_tensor


class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None):
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        # Set device if not provided
        if device is None:
            device = next(self.unet.parameters()).device

        # Handle class conditioning
        if classes is not None:
            if isinstance(classes, int):
                classes = torch.tensor([classes] * batch_size, device=device)
            elif isinstance(classes, list):
                classes = torch.tensor(classes, device=device)
            
            # Get class embeddings
            class_embeds = self.class_embedder(classes)
            if guidance_scale is not None and guidance_scale > 1:
                # For classifier free guidance, we need to do two forward passes.
                # Here we create a encoder_hidden_states, which is twice the batch size
                uncond_embeds = self.class_embedder(torch.tensor([self.class_embedder.num_classes] * batch_size, device=device))
                class_embeds = torch.cat([uncond_embeds, class_embeds])

        # Create initial noise
        latents = randn_tensor(
            (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size),
            generator=generator,
            device=device,
        )

        # Scale the latents if using VAE
        if self.vae is not None:
            latents = latents * 0.18215

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device)

        # Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            # Expand latents for classifier free guidance
            if guidance_scale is not None and guidance_scale > 1:
                latent_model_input = torch.cat([latents] * 2)
                timestep_batch = torch.cat([t.unsqueeze(0)] * 2)
            else:
                latent_model_input = latents
                timestep_batch = t.unsqueeze(0)

            # Predict noise residual
            if classes is not None and guidance_scale is not None and guidance_scale > 1:
                noise_pred = self.unet(latent_model_input, timestep_batch, class_embeds)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                c = class_embeds if classes is not None else None
                noise_pred = self.unet(latent_model_input, timestep_batch, c)

            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator)

        # Decode latents if using VAE
        if self.vae is not None:
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents)
        else:
            image = latents

        # Ensure images are in [0, 1] range
        image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)

        # Add proper denormalization for visualization
        image = (image * 255).round().clamp(0, 255)

        # Convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return image
