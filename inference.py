import os
import sys
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint
from train import parse_args

logger = get_logger(__name__)

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=args.unet_ch,
    ).to(device)

    # Initialize scheduler
    scheduler_class = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    ).to(device)

    # Initialize VAE if using latent DDPM
    vae = None
    if args.latent_ddpm:
        vae = VAE().to(device)
        vae.init_from_ckpt("pretrained/model.ckpt")
        vae.eval()

    # Initialize class embedder if using CFG
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(
            num_classes=args.num_classes,
            embed_dim=args.unet_ch,
        ).to(device)

    # Load checkpoint
    load_checkpoint(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
        checkpoint_path=args.ckpt
    )

    # Initialize pipeline
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
    )

    # Initialize metrics
    fid = FrechetInceptionDistance(normalize=True).to(device)
    is_score = InceptionScore(normalize=True).to(device)

    logger.info("Generating images...")
    all_images = []
    
    if args.use_cfg:
        # Generate images for each class
        images_per_class = 50
        for class_idx in tqdm(range(args.num_classes)):
            gen_images = pipeline(
                batch_size=images_per_class,
                num_inference_steps=args.num_inference_steps,
                classes=class_idx,
                guidance_scale=args.cfg_guidance_scale,
                device=device
            )
            all_images.extend(gen_images)
    else:
        # Generate unconditional images
        total_images = 5000
        batch_size = 50
        
        for _ in tqdm(range(0, total_images, batch_size)):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                device=device
            )
            all_images.extend(gen_images)

    # Convert PIL images to tensors for metric computation
    generated_tensors = []
    for img in all_images:
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        generated_tensors.append(img_tensor)
    generated_tensors = torch.stack(generated_tensors).to(device)

    # Update metrics with generated images
    fid.update(generated_tensors, real=False)
    is_score.update(generated_tensors)

    # Load and process validation images
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "../val"), transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Update metrics with real images
    for real_images, _ in tqdm(val_loader, desc="Processing real images"):
        real_images = real_images.to(device)
        fid.update(real_images, real=True)

    # Compute final metrics
    fid_score = fid.compute()
    is_mean, is_std = is_score.compute()

    logger.info(f"FID Score: {fid_score:.2f}")
    logger.info(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")

    # Save results
    results = {
        "fid": float(fid_score),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
    }
    
    # Save metrics to file
    output_dir = os.path.dirname(args.ckpt)
    with open(os.path.join(output_dir, "metrics.yaml"), "w") as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main()
