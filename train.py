### train.py ###
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

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import (
    seed_everything,
    init_distributed_device,
    is_primary,
    AverageMeter,
    str2bool,
    save_checkpoint,
)

from torch.cuda.amp import autocast, GradScaler

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    # config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ddpm.yaml",
        help="config file used to specify parameters",
    )

    # data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/imagenet100_128x128/train",
        help="data folder",
    )
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument(
        "--num_classes", type=int, default=100, help="number of classes in dataset"
    )

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument(
        "--output_dir", type=str, default="experiments", help="output folder"
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="none",
        choices=["fp16", "bf16", "fp32", "none"],
        help="mixed precision",
    )

    # ddpm
    parser.add_argument(
        "--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=200, help="ddpm inference timesteps"
    )
    parser.add_argument(
        "--beta_start", type=float, default=0.0002, help="ddpm beta start"
    )
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", help="ddpm beta schedule"
    )
    parser.add_argument(
        "--variance_type", type=str, default="fixed_small", help="ddpm variance type"
    )
    parser.add_argument(
        "--prediction_type", type=str, default="epsilon", help="ddpm epsilon type"
    )
    parser.add_argument(
        "--clip_sample",
        type=str2bool,
        default=True,
        help="whether to clip sample at each step of reverse process",
    )
    parser.add_argument(
        "--clip_sample_range", type=float, default=1.0, help="clip sample range"
    )

    # unet
    parser.add_argument(
        "--unet_in_size", type=int, default=128, help="unet input image size"
    )
    parser.add_argument(
        "--unet_in_ch", type=int, default=3, help="unet input channel size"
    )
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument(
        "--unet_ch_mult",
        type=int,
        default=[1, 2, 2, 2],
        nargs="+",
        help="unet channel multiplier",
    )
    parser.add_argument(
        "--unet_attn",
        type=int,
        default=[1, 2, 3],
        nargs="+",
        help="unet attantion stage index",
    )
    parser.add_argument(
        "--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks"
    )
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")

    # vae
    parser.add_argument(
        "--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm"
    )

    # cfg
    parser.add_argument(
        "--use_cfg",
        type=str2bool,
        default=False,
        help="use cfg for conditional (latent) ddpm",
    )
    parser.add_argument(
        "--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference"
    )

    # ddim sampler for inference
    parser.add_argument(
        "--use_ddim",
        type=str2bool,
        default=False,
        help="use ddim sampler for inference",
    )

    # checkpoint path for inference
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint path for inference"
    )

    # gradient accumulation
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating",
    )

    # first parse of command-line args to check for config file
    args = parser.parse_args()

    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args

def train_epoch(args, epoch, unet, scheduler, vae, class_embedder, train_loader, 
                optimizer, scaler, device, wandb_logger):
    """Single epoch training function"""
    unet.train()
    scheduler.train()
    loss_meter = AverageMeter()
    
    progress_bar = tqdm(range(len(train_loader)), disable=not is_primary(args))
    
    for step, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        
        # Move data to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Handle VAE encoding if using latent DDPM
        if vae is not None:
            with torch.no_grad():
                images = vae.encode(images).sample()
                images = images * 0.18215
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get class embeddings if using CFG
        if class_embedder is not None:
            class_emb = class_embedder(labels)
        else:
            class_emb = None
            
        # Sample noise and timesteps
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, args.num_train_timesteps, (batch_size,), device=device)
        
        # Add noise to images
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        
        # Forward pass with mixed precision
        with autocast(enabled=(args.mixed_precision in ["fp16", "bf16"])):
            model_pred = unet(noisy_images, timesteps, class_emb)
            
            if args.prediction_type == "epsilon":
                target = noise
            
            loss = F.mse_loss(model_pred, target)
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
                optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item() * args.gradient_accumulation_steps)
        progress_bar.update(1)
        
        # Logging
        if step % 100 == 0 and is_primary(args):
            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{len(train_loader)}, "
                f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f} ({loss_meter.avg:.4f})"
            )
            if wandb_logger:
                wandb_logger.log({"train/loss": loss_meter.avg})
    
    return loss_meter.avg

def validate(args, epoch, pipeline, device, wandb_logger):
    """Validation function that generates and logs sample images"""
    pipeline.unet.eval()
    
    with torch.no_grad():
        # Generate samples
        if args.use_cfg:
            # Sample random classes
            classes = torch.randint(0, args.num_classes, (4,), device=device)
            samples = pipeline(
                batch_size=4,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                device=device
            )
        else:
            samples = pipeline(
                batch_size=4,
                num_inference_steps=args.num_inference_steps,
                device=device
            )
        
        # Ensure samples are properly normalized
        samples = [(img.resize((args.image_size, args.image_size)) if img.size != (args.image_size, args.image_size) else img)
                  for img in samples]
        
        # Create image grid with proper normalization
        grid = Image.new('RGB', (4 * args.image_size, args.image_size))
        for idx, img in enumerate(samples):
            grid.paste(img, (idx * args.image_size, 0))
        
        # Log to wandb with proper caption
        if is_primary(args) and wandb_logger:
            wandb_logger.log({
                "samples": wandb.Image(grid, caption=f"Epoch {epoch+1}"),
                "current_timestep": args.num_inference_steps,
            })
        
        return grid

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up distributed training
    device = init_distributed_device(args)
    
    # Set random seed
    seed_everything(args.seed)
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    train_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    
    # Create models
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
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
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
    
    # Set up distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.local_rank], output_device=args.local_rank
        )
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.local_rank], output_device=args.local_rank
            )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Set up mixed precision training
    scaler = GradScaler() if args.mixed_precision in ["fp16", "bf16"] else None
    
    # Initialize wandb
    wandb_logger = None
    if is_primary(args):
        wandb_logger = wandb.init(project="ddpm", name=args.run_name, config=vars(args))
    
    # Create pipeline for validation
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(
            args, epoch, unet, scheduler, vae, class_embedder,
            train_loader, optimizer, scaler, device, wandb_logger
        )
        
        # Validate and generate samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            validate(args, epoch, pipeline, device, wandb_logger)
        
        # Save checkpoint
        if is_primary(args):
            save_checkpoint(
                unet=unet.module if args.distributed else unet,
                scheduler=scheduler,
                vae=vae,
                class_embedder=class_embedder.module if args.distributed and class_embedder else class_embedder,
                optimizer=optimizer,
                epoch=epoch,
                save_dir=os.path.join(args.output_dir, args.run_name, "checkpoints"),
            )
    
    # Clean up
    if wandb_logger:
        wandb_logger.finish()

if __name__ == "__main__":
    main()
