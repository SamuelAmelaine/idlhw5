import torch
import torch.nn as nn
from contextlib import contextmanager

from .vae_modules import Encoder, Decoder
from .vae_distributions import DiagonalGaussianDistribution


class VAE(nn.Module):
    def __init__(
        self,
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1,2,4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        },
        embed_dim=4,
    ):
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=ddconfig["in_channels"],
            ch=ddconfig["ch"],
            out_ch=ddconfig["out_ch"],
            num_res_blocks=ddconfig["num_res_blocks"],
            z_channels=ddconfig["z_channels"],
            ch_mult=ddconfig["ch_mult"],
            resolution=ddconfig["resolution"],
            double_z=ddconfig["double_z"],
            attn_resolutions=ddconfig["attn_resolutions"],
        )
        self.decoder = Decoder(
            in_channels=ddconfig["in_channels"],
            ch=ddconfig["ch"],
            out_ch=ddconfig["out_ch"],
            num_res_blocks=ddconfig["num_res_blocks"],
            z_channels=ddconfig["z_channels"],
            ch_mult=ddconfig["ch_mult"],
            resolution=ddconfig["resolution"],
            double_z=ddconfig["double_z"],
            attn_resolutions=ddconfig["attn_resolutions"],
        )
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    @torch.no_grad()
    def encode(self, x):
        """Transform images into a sampled vector using VAE encoder"""
        # Pass through encoder to get latent representation
        h = self.encoder(x)
        # Get moments (mean, logvar) from latent representation
        moments = self.quant_conv(h)
        # Sample from Gaussian using reparameterization trick
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    @torch.no_grad()
    def decode(self, z):
        """Reconstruct images from latent vectors"""
        # Transform latent vector through post quantization convolution
        z = self.post_quant_conv(z)
        # Decode to image space
        dec = self.decoder(z)
        return dec

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(keys)
