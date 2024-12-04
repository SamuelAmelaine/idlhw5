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
from time import time

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

    # Add gradient checkpointing argument
    parser.add_argument(
        "--use_gradient_checkpointing",
        type=str2bool,
        default=False,
        help="use gradient checkpointing to save memory",
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
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    progress_bar = tqdm(range(len(train_loader)), disable=not is_primary(args))
    
    optimizer.zero_grad()  # Move outside the loop
    
    start_time = time()
    images_processed = 0
    
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
        with torch.amp.autocast('cuda', enabled=(args.mixed_precision in ["fp16", "bf16"])):
            model_pred = unet(noisy_images, timesteps, class_emb)
            
            if args.prediction_type == "epsilon":
                target = noise
            
            loss = F.mse_loss(model_pred, target)
            loss = loss / args.gradient_accumulation_steps
        
        # Add gradient check only on first batch of first epoch
        if epoch == 0 and step == 0:  
            with torch.no_grad():  # Don't accumulate gradients for check
                has_gradients = any(p.requires_grad for p in unet.parameters())
                if not has_gradients:
                    raise RuntimeError("No parameters have requires_grad=True!")
        
        # Regular backward pass
        if scaler:
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
                    if wandb_logger:
                        wandb_logger.log({"train/gradient_norm": grad_norm})
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
                    if wandb_logger:
                        wandb_logger.log({"train/gradient_norm": grad_norm})
                optimizer.step()
                optimizer.zero_grad()
        
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
        
        images_processed += images.shape[0]
        
        if step % 100 == 0 and is_primary(args):
            elapsed = time() - start_time
            images_per_sec = images_processed / elapsed
            logger.info(f"Training speed: {images_per_sec:.2f} images/second")
            
            if wandb_logger:
                wandb_logger.log({
                    "train/loss": loss_meter.avg,
                    "train/images_per_second": images_per_sec,
                    "train/gpu_memory_used": torch.cuda.max_memory_allocated() / 1024**3
                })
    
    return loss_meter.avg

def validate(args, epoch, pipeline, device, wandb_logger):
    """Validation function that generates and logs sample images"""
    pipeline.unet.eval()
    
    with torch.no_grad():
        # Generate multiple samples with different noise levels
        timesteps_to_test = [args.num_inference_steps, args.num_inference_steps // 2, args.num_inference_steps // 4]
        all_samples = []
        
        for num_inference_steps in timesteps_to_test:
            samples = pipeline(
                batch_size=4,
                num_inference_steps=num_inference_steps,  # Try different step counts
                device=device
            )
            
            # Ensure samples are properly normalized
            samples = [(img.resize((args.image_size, args.image_size)) if img.size != (args.image_size, args.image_size) else img)
                      for img in samples]
            all_samples.extend(samples)
        
        # Create image grid with proper normalization
        grid = Image.new('RGB', (4 * args.image_size, len(timesteps_to_test) * args.image_size))
        for idx, img in enumerate(all_samples):
            row = idx // 4
            col = idx % 4
            grid.paste(img, (col * args.image_size, row * args.image_size))
        
        # Log to wandb with proper caption
        if is_primary(args) and wandb_logger:
            wandb_logger.log({
                "samples": wandb.Image(grid, caption=f"Epoch {epoch+1}\nRows: {timesteps_to_test} steps"),
                "current_timestep": args.num_inference_steps,
            })
        
        return grid

def main():
    # Add these lines at the start of main()
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions
    
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
        transforms.Resize((args.image_size, args.image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.ConvertImageDtype(torch.float16),  # Pre-convert to fp16
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
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    # Create models with explicit requires_grad
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
    
    # Explicitly set requires_grad
    for param in unet.parameters():
        param.requires_grad = True
    
    # Print parameter count and gradient status
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Verify gradients are enabled
    for name, param in unet.named_parameters():
        if not param.requires_grad:
            logger.warning(f"Parameter {name} has requires_grad=False")
    
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
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision in ["fp16", "bf16"] else None
    
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
    
    # After creating the UNet model
    if args.use_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # At the start of main()
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cudnn.benchmark = True
    
    # Add at the start of main()
    torch.backends.cuda.max_memory_allocated = 0
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Verify gradients before training
    from utils.misc import verify_model_gradients
    grad_status = verify_model_gradients(unet, F.mse_loss, device)
    
    # Log gradient status
    if is_primary(args):
        logger.info("Gradient status check:")
        for name, status in grad_status.items():
            logger.info(f"{name}:")
            logger.info(f"  requires_grad: {status['requires_grad']}")
            logger.info(f"  grad_is_none: {status['grad_is_none']}")
            logger.info(f"  grad_norm: {status['grad_norm']:.6f}")
    
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
