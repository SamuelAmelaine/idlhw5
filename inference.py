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

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()

    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
    )
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    # TODO: ddpm scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.predictor_type,
    )
    # vae
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt("pretrained/model.ckpt")
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch
        )  # TODO: check if this is correct,

    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # scheduler
    if args.use_ddim:
        scheduler_class = DDIMScheduler
    else:
        scheduler_class = DDPMScheduler

    # TODO: ddpm scheduler
    scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.predictor_type,
    )

    # load checkpoint
    load_checkpoint(
        unet,
        scheduler,
        vae=vae,
        class_embedder=class_embedder,
        checkpoint_path=args.ckpt,
    )

    # TODO: pipeline
    pipeline = DDPMPipeline(
        unet=unet, scheduler=scheduler, vae=vae, class_embedder=class_embedder
    )

    logger.info("***** Running Infrence *****")

    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class
    all_images = []
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                # guidance_scale=args.cfg_guidance_scale,  # Should be here but not in ddpm.yaml?
                generator=generator,
                device=device,
            )
            all_images.append(gen_images)
    else:
        # generate 5000 images
        batch_size = 50  # Process in batches of 50
        for _ in tqdm(range(0, 5000, batch_size)):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            all_images.append(gen_images)

    # TODO: load validation images as reference batch
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics

    from torchmetrics.image.fid import FrechetInceptionDistance, InceptionScore
    from utils.dist import is_primary

    fid = FrechetInceptionDistance(normalize=True).to(device)
    is_score = InceptionScore(normalize=True).to(device)

    # TODO: compute FID and IS
    # First pass real images
    for real_images, _ in tqdm(val_loader, desc="Processing real images"):
        real_images = real_images.to(device)
        fid.update(real_images, real=True)
        is_score.update(real_images)

    # Then pass generated images
    for gen_batch in tqdm(all_images, desc="Processing generated images"):
        gen_batch = torch.stack([transforms.ToTensor()(img) for img in gen_batch]).to(
            device
        )
        gen_batch = (gen_batch * 2) - 1  # Scale to [-1, 1]
        fid.update(gen_batch, real=False)
        is_score.update(gen_batch)

    fid_score = fid.compute()
    is_mean, is_std = is_score.compute()

    logger.info(f"FID Score: {fid_score:.2f}")
    logger.info(f"IS Score: {is_mean:.2f} ± {is_std:.2f}")

    if is_primary(args):
        wandb.log(
            {
                "fid": fid_score,
                "inception_score_mean": is_mean,
                "inception_score_std": is_std,
            }
        )


if __name__ == "__main__":
    main()
