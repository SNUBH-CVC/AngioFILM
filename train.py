# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP with Flow Matching.
"""

import argparse
import math
from copy import deepcopy
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import create_alternating_dataset
from model import get_model
from tokenizer import get_frozen_tokenizer
from transport import create_transport
from utils import (cleanup, clip_grad_norm_, create_logger, create_tensorboard,
                   get_experiment_dir, get_resume_experiment_dir,
                   load_checkpoint, requires_grad, setup_distributed,
                   update_ema, write_tensorboard)

# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    seed = args.training.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}."
    )
    if rank == 0:
        results_dir = Path(args.training.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Setup experiment folder: 
        # If resuming from checkpoint, reuse existing experiment directory
        if hasattr(args.training, 'resume_from_checkpoint') and args.training.resume_from_checkpoint:
            resume_checkpoint_path = args.training.resume_from_checkpoint
            experiment_dir = get_resume_experiment_dir(resume_checkpoint_path)
            if experiment_dir:
                logger_msg = f"Resuming experiment in existing directory: {experiment_dir}"
            else:
                raise ValueError(f"Could not determine experiment directory from checkpoint path: {resume_checkpoint_path}")
        else:
            # Create new experiment directory
            experiment_dir = get_experiment_dir(results_dir, args)
            logger_msg = f"Experiment directory created at {experiment_dir}"
        
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, experiment_dir / "config.yaml")
        logger.info(logger_msg)
    else:
        logger = create_logger(None)
        tb_writer = None

    tokenizer = get_frozen_tokenizer(args.training.tokenizer_name, device)
    logger.info(f"Tokenizer loaded: {args.training.tokenizer_name}")

    # Create transport for flow matching
    transport = create_transport(
        args.transport.path_type,
        args.transport.prediction,
        args.transport.loss_weight,
        args.transport.train_eps,
        args.transport.sample_eps
    )
    logger.info("Using Flow Matching method")

    # Create model
    assert args.model.image_size % tokenizer.img_size_divisor == 0, (
        f"Image size must be divisible by {tokenizer.img_size_divisor} (for {args.training.tokenizer_name})."
    )
    args.model.latent_size = args.model.image_size // 8
    model = get_model(
        args.model.name, 
        args.model.latent_size, 
        tokenizer.latent_dim,
        args.model.num_frames, 
        args.model.mask_cond_prob, 
        args.model.ecg_mask_cond_prob, 
        args.model.use_ecg, 
        args.model.ecg_signal_len
    )

    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # set distributed training
    model = DDP(
        model.to(device), device_ids=[local_rank]
    )  

    # Apply torch.compile if enabled
    if args.system.use_compile:
        logger.info("Applying torch.compile to the model...")
        try:
            # Use different compilation modes based on config
            compile_mode = getattr(args.system, 'compile_mode', 'default')
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"torch.compile applied successfully with mode: {compile_mode}")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Falling back to regular model.")
            # Model remains uncompiled if compilation fails

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.base_lr, weight_decay=0, betas=args.training.betas)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.system.mixed_precision else None

    # Setup data:
    dataset = create_alternating_dataset(
        args.dataset.train_path,
        image_size=args.model.image_size,
        num_frames=args.model.num_frames,
        use_ecg=args.model.use_ecg,
        ecg_signal_len=args.model.ecg_signal_len,
        test_mode=False
    )
    sampler = DistributedSampler(
        dataset,
        rank=rank,
        shuffle=True,
        seed=args.training.global_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.training.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.dataset.train_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name=args.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.training.lr_warmup_steps,
        num_training_steps=args.training.max_train_steps,
    )

    # Initialize training state
    train_steps = 0
    start_epoch = 0
    
    # Load checkpoint if resuming - ALL PROCESSES MUST LOAD
    if hasattr(args.training, 'resume_from_checkpoint') and args.training.resume_from_checkpoint:
        checkpoint_path = Path(args.training.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # ALL processes load the checkpoint to ensure parameter synchronization
        train_steps, start_epoch = load_checkpoint(
            checkpoint_path, model, ema, optimizer, lr_scheduler, scaler, device
        )
        
        # Ensure all processes are synchronized after loading
        dist.barrier()
        logger.info(f"All processes resumed from train_steps={train_steps}, start_epoch={start_epoch}")

        # Skip EMA initialization since it's already loaded from checkpoint
        skip_ema_init = True
    else:
        skip_ema_init = False

    # Prepare models for training:
    if not skip_ema_init:
        update_ema(
            ema, model.module, decay=0
        )  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    start_time = time()
    running_loss = 0

    num_train_epochs = math.ceil(args.training.max_train_steps / len(dataloader))

    for epoch in range(start_epoch, num_train_epochs):
        sampler.set_epoch(epoch)

        for batch in dataloader:
            train_steps += 1

            x1 = batch["frames"].to(device, non_blocking=True)
            cond_frames = batch["cond_frames"].to(device, non_blocking=True)

            # Map input images to latent space + normalize latents:
            with torch.no_grad():
                b, _, _, _, _ = x1.shape
                x1 = rearrange(x1, "b f c h w -> (b f) c h w").contiguous()
                x1 = tokenizer.encode(x1)
                x1 = rearrange(x1, "(b f) c h w -> b f c h w", b=b).contiguous()

                cond_frames = rearrange(cond_frames, "b f c h w -> (b f) c h w").contiguous()
                cond_frames = tokenizer.encode(cond_frames)
                cond_frames = rearrange(cond_frames, "(b f) c h w -> b f c h w", b=b).contiguous()

            model_kwargs = dict(y_image=cond_frames)
            if args.model.use_ecg:
                model_kwargs["y_ecg"] = batch["cond_ecg"].to(device)

            # Flow matching training loss
            with autocast(device_type=device.type, enabled=args.system.mixed_precision):
                # Determine x0 based on config
                if args.transport.x0_type == "conditioning":
                    # Use conditional frames as x0
                    x0 = cond_frames[:, :-1] 
                elif args.transport.x0_type == "gaussian":
                    x0 = None  # Use random Gaussian noise as x0
                else:
                    raise ValueError(f"Unknown x0_type: {args.transport.x0_type}")
                
                loss_dict = transport.training_losses(model, x1, x0, model_kwargs)
                loss = loss_dict["loss"].mean()
                
            # Backward pass with gradient scaling
            if args.system.mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()

            if args.system.mixed_precision and scaler is not None:
                scaler.unscale_(optimizer)
                gradient_norm = clip_grad_norm_(
                    model.module.parameters(), args.training.clip_max_norm, clip_grad=True
                )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                gradient_norm = clip_grad_norm_(
                    model.module.parameters(), args.training.clip_max_norm, clip_grad=True
                )
                
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            update_ema(ema, model.module)

            if train_steps % args.training.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = args.training.log_every / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / args.training.log_every, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                write_tensorboard(tb_writer, "Train Loss", avg_loss, train_steps)
                write_tensorboard(
                    tb_writer, "Gradient Norm", gradient_norm, train_steps
                )
                write_tensorboard(
                    tb_writer, "Learning Rate", lr_scheduler.get_last_lr()[0], train_steps
                )

                start_time = time()
                running_loss = 0

            if train_steps % args.training.ckpt_every == 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "train_steps": train_steps,
                        "epoch": epoch,
                    }
                    if scaler is not None:
                        checkpoint["scaler"] = scaler.state_dict()
                    checkpoint_path = checkpoint_dir / f"{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps >= args.training.max_train_steps:
                break

        if train_steps >= args.training.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, 
                       help="Path to config file (ignored if --ckpt_path is provided)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                       help="Path to checkpoint for resuming training. Config will be auto-detected from checkpoint directory.")
    args = parser.parse_args()
    
    if args.ckpt_path:
        # Resume mode: auto-detect config from checkpoint path
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        # Extract experiment directory (checkpoint path structure: experiment_dir/checkpoints/xxxxx.pt)
        if ckpt_path.parent.name == "checkpoints":
            experiment_dir = ckpt_path.parent.parent
            config_path = experiment_dir / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            print(f"Resume mode: Loading config from {config_path}")
            print(f"Resume mode: Loading checkpoint from {ckpt_path}")
            config = OmegaConf.load(config_path)
            
            # Set resume checkpoint path in config
            config.training.resume_from_checkpoint = str(ckpt_path)
        else:
            raise ValueError(f"Invalid checkpoint path structure. Expected: experiment_dir/checkpoints/xxxxx.pt, got: {ckpt_path}")
    else:
        # Normal mode: use provided config
        config = OmegaConf.load(args.config)
    
    main(config)
