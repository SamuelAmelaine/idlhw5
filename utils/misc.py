from typing import List, Optional, Tuple, Union

import os 
import argparse
import random 
import numpy as np

import torch 
import torch.nn.functional as F


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`."""
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def verify_model_gradients(model, criterion, device):
    """Verify model gradients using a dummy forward pass."""
    model.train()
    
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create dummy inputs with proper timestep range
        batch_size = 2
        channels = model.input_ch if hasattr(model, 'input_ch') else 3
        size = model.input_size if hasattr(model, 'input_size') else 64
        
        # Ensure inputs are on the correct device and dtype
        x = torch.randn(batch_size, channels, size, size, device=device)
        # Use model's num_train_timesteps if available
        max_timesteps = model.T if hasattr(model, 'T') else 1000
        t = torch.randint(0, max_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x)
        
        # Forward pass with error handling
        with torch.enable_grad():
            try:
                pred = model(x, t)
                loss = criterion(pred, noise)
                loss.backward()
            except RuntimeError as e:
                print(f"Error during forward/backward pass: {e}")
                return None
        
        # Check gradients
        grad_status = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_status[name] = {
                    'requires_grad': param.requires_grad,
                    'grad_is_none': False,
                    'grad_norm': torch.norm(param.grad).item()
                }
            else:
                grad_status[name] = {
                    'requires_grad': param.requires_grad,
                    'grad_is_none': True,
                    'grad_norm': 0.0
                }
        
        return grad_status
        
    except Exception as e:
        print(f"Error in gradient verification: {e}")
        return None