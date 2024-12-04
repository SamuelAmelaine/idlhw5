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
    verify_model_gradients,
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
    unet.train()
    loss_meter = AverageMeter()
    progress_bar = tqdm(range(len(train_loader)), disable=not is_primary(args))
    progress_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
    
    # Initialize timing and processing counters
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
                if args.mixed_precision in ["fp16", "bf16"]:
                    with torch.cuda.amp.autocast():
                        images = vae.encode(images).sample()
                else:
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
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        # Update metrics and logging
        loss_meter.update(loss.item() * args.gradient_accumulation_steps)
        images_processed += batch_size
        progress_bar.update(1)
        
        # More frequent and detailed logging
        if step % 50 == 0 and is_primary(args):  # Log every 50 steps
            elapsed = time() - start_time
            images_per_sec = images_processed / elapsed
            
            log_str = (f"[Epoch {epoch+1}/{args.num_epochs}][Step {step}/{len(train_loader)}] "
                      f"Loss: {loss_meter.avg:.4f} | "
                      f"Images/sec: {images_per_sec:.1f}")
            logger.info(log_str)
            
            if wandb_logger:
                wandb_logger.log({
                    "train/loss": loss_meter.avg,
                    "train/images_per_second": images_per_sec,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch + 1,
                    "train/step": step,
                })
    
    # End of epoch logging
    if is_primary(args) and wandb_logger:
        wandb_logger.log({
            "epoch": epoch + 1,
            "epoch_loss": loss_meter.avg,
            "epoch_images_per_second": images_processed / (time() - start_time),
        })
    
    return loss_meter.avg

def validate(args, epoch, pipeline, device, wandb_logger):
    """Validation with better image grid visualization"""
    pipeline.unet.eval()
    
    with torch.no_grad():
        # Generate a larger batch of images for better visualization
        n_samples = 16  # 4x4 grid
        samples = pipeline(
            batch_size=n_samples,
            num_inference_steps=args.num_inference_steps,
            device=device
        )
        
        # Create a grid of images
        from torchvision.utils import make_grid
        from PIL import Image
        import numpy as np
        
        # Convert PIL images to tensors
        sample_tensors = []
        for img in samples:
            # Convert PIL image to tensor
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            sample_tensors.append(img_tensor)
        
        # Create grid
        grid_tensor = make_grid(torch.stack(sample_tensors), nrow=4)
        
        # Convert to PIL for wandb
        grid_image = Image.fromarray(
            (grid_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        
        # Log to wandb with more descriptive caption
        if is_primary(args) and wandb_logger:
            wandb_logger.log({
                "generated_samples": wandb.Image(
                    grid_image, 
                    caption=f"Epoch {epoch+1} | Step {args.num_inference_steps} steps"
                ),
                "current_epoch": epoch + 1,
            })
        
        return grid_image

def main():
    # CUDA setup and memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocator settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
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
        vae = VAE(
            ddconfig=args.vae_config,
            embed_dim=args.unet_in_ch
        ).to(device)
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
    
    # Verify gradients before training
    if is_primary(args):
        logger.info("Verifying gradients...")
        grad_status = verify_model_gradients(unet, F.mse_loss, device)
        
        if grad_status is not None:
            logger.info("Gradient check successful")
            # Only log a summary of gradient status
            num_grad_none = sum(1 for status in grad_status.values() if status['grad_is_none'])
            num_params = len(grad_status)
            logger.info(f"Parameters with gradients: {num_params - num_grad_none}/{num_params}")
        else:
            logger.warning("Gradient check failed, but continuing with training")
    
    # Add learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_lr
    )
    
    # Simplified training loop
    for epoch in range(args.num_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            args, epoch, unet, scheduler, vae, class_embedder,
            train_loader, optimizer, scaler, device, wandb_logger
        )
        
        # Validate more frequently (every epoch)
        if is_primary(args):
            validate(args, epoch, pipeline, device, wandb_logger)
            
            # Log epoch summary
            if wandb_logger:
                wandb_logger.log({
                    "epoch": epoch + 1,
                    "epoch_loss": train_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                })
        
        # Save checkpoint every 5 epochs
        if is_primary(args) and (epoch + 1) % 5 == 0:
            save_checkpoint(
                unet=unet.module if args.distributed else unet,
                scheduler=scheduler,
                vae=vae,
                class_embedder=class_embedder.module if args.distributed and class_embedder else class_embedder,
                optimizer=optimizer,
                epoch=epoch,
                save_dir=os.path.join(args.output_dir, args.run_name, "checkpoints"),
            )
        
        lr_scheduler.step()
    
    # Clean up
    if wandb_logger:
        wandb_logger.finish()

if __name__ == "__main__":
    main()
