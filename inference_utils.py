#################################################################################
#                             Testing  Utils                                    #
#################################################################################

import math
import torch


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros(
        (t, (padding + h) * nrow + padding, (padding + w) * ncol + padding, c),
        dtype=torch.uint8,
    )

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r : start_r + h, start_c : start_c + w] = video[i]

    return video_grid


#################################################################################
#                             Sampling Utils                                    #
#################################################################################

import os

import cv2
import imageio.v3 as iio
import torch
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import get_model
from tokenizer import get_frozen_tokenizer
from transport import Sampler, create_transport



def find_model(model_name):
    """
    Finds a pre-trained Latte model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), (
        f"Could not find Latte checkpoint at {model_name}"
    )
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)

    if "ema" in checkpoint:  # supports checkpoints from train.py
        print("Using Ema!")
        checkpoint = checkpoint["ema"]
    else:
        print("Using model!")
        checkpoint = checkpoint["model"]
    return checkpoint


def setup_models(config_args, ckpt_path, device):
    """
    Initialize and load the main model, transport process, and tokenizer.

    Args:
        config_args: Configuration parameters from config file.
        ckpt_path: Path to the checkpoint file.
        device: The torch device (e.g., "cuda:0").

    Returns:
        tuple: (model, transport, tokenizer) loaded onto the device.
    """
    tokenizer = get_frozen_tokenizer(config_args.training.tokenizer_name, device)
    print(f"Tokenizer loaded: {config_args.training.tokenizer_name}")

    # Create transport for flow matching
    transport = create_transport(
        config_args.transport.path_type,
        config_args.transport.prediction,
        config_args.transport.loss_weight,
        config_args.transport.train_eps,
        config_args.transport.sample_eps
    )
    print(f"Using Flow Matching method with {config_args.transport.path_type} path type")

    # Main model
    model = get_model(
        config_args.model.name,
        config_args.model.latent_size,
        tokenizer.latent_dim,
        config_args.model.num_frames,
        config_args.model.mask_cond_prob,
        config_args.model.ecg_mask_cond_prob,
        config_args.model.use_ecg,
        config_args.model.ecg_signal_len
    ).to(device)

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)  # set strict=False for use_ecg=False
    model.eval()

    return model, transport, tokenizer


def generate_sample_for_clip(video_cond, ecg_cond, model, transport, tokenizer, config_args, cfg_scale, num_sampling_steps,
                           save_intermediate=False, intermediate_steps=None):
    """
    Generate samples for a single video clip using the flow matching model.

    Args:
        video_cond (torch.Tensor): Conditioning clip tensor of shape (B, F+1, C, H, W).
        ecg_cond (torch.Tensor): ECG conditioning clip tensor of shape (B, F, C, H, W).
        model: The main generative model.
        transport: Transport process object.
        tokenizer: Tokenizer for encoding and decoding.
        config_args: Configuration parameters from config file.
        cfg_scale: Classifier-free guidance scale.
        num_sampling_steps: Number of sampling steps.
        save_intermediate (bool): Whether to save intermediate results.
        intermediate_steps (list): Specific steps to save intermediate results.

    Returns:
        tuple: (final_decoded_samples, intermediate_results_dict) where
               final_decoded_samples has shape (B, F, C, H, W) and
               intermediate_results_dict contains decoded intermediate results.
    """
    batch_size = video_cond.shape[0]

    # Initialize x0 based on config
    if config_args.transport.x0_type == "conditioning":
        # Use even frames as x0 for flow matching (even → odd)
        # video_cond shape: (B, F+1, C, H, W), we need first F frames
        even_frames = video_cond[:, :-1]  # (B, F, C, H, W) - exclude last frame
        with torch.no_grad():
            even_frames_reshaped = rearrange(even_frames, 'b f c h w -> (b f) c h w').contiguous()
            z = tokenizer.encode(even_frames_reshaped)
            z = rearrange(z, '(b f) c h w -> b f c h w', b=batch_size).contiguous()
    elif config_args.transport.x0_type == "gaussian":
        # Use random Gaussian noise as x0
        latent_size = config_args.model.latent_size
        z = torch.randn(batch_size, config_args.model.num_frames, tokenizer.latent_dim, latent_size, latent_size, device=video_cond.device)
    else:
        raise ValueError(f"Unknown x0_type: {config_args.transport.x0_type}")

    with torch.no_grad():
        video_cond_reshaped = rearrange(video_cond, 'b f_p1 c h w -> (b f_p1) c h w').contiguous()
        latent = tokenizer.encode(video_cond_reshaped)
        video_condition = rearrange(latent, '(b f_p1) c h w -> b f_p1 c h w', b=batch_size).contiguous()

    use_cfg = config_args.model.mask_cond_prob > 0.0
    model_kwargs = dict(y_image=video_condition, y_ecg=ecg_cond)
    if use_cfg:
        sample_fn = model.forward_with_cfg
        model_kwargs.update(dict(cfg_scale=cfg_scale))
    else:
        assert config_args.model.mask_cond_prob == 0.0, "mask_cond_prob must be 0.0 when use_cfg is False"
        sample_fn = model.forward

    # Determine which steps to save
    steps_to_save = set()
    if save_intermediate:
        if intermediate_steps:
            steps_to_save.update(intermediate_steps)

    # Auto-select sampling method based on path type
    transport_sampler = Sampler(transport)

    # Flow Matching (Linear/GVP/VP) uses ODE Euler
    sample_fn_ode = transport_sampler.sample_ode(sampling_method='euler', num_steps=num_sampling_steps)
    sampling_results = sample_fn_ode(z, sample_fn, **model_kwargs)  # (num_sampling_steps, B, F, C, H, W)

    # Add initial gaussian noise z to the beginning of all_samples
    z_expanded = z.unsqueeze(0)  # (1, B, F, C, H, W)
    all_samples = torch.cat([z_expanded, sampling_results], dim=0)  # (num_sampling_steps+1, B, F, C, H, W)

    # Get final samples (last index after adding initial noise)
    final_samples = all_samples[-1]

    # Collect intermediate results if requested
    intermediate_results = {}
    if save_intermediate and steps_to_save:
        for step_idx, step_samples in enumerate(all_samples):
            current_step = step_idx
            if current_step in steps_to_save:
                # Decode intermediate samples
                with torch.no_grad():
                    b, f, c, h, w = step_samples.shape
                    step_samples_reshaped = rearrange(step_samples, 'b f c h w -> (b f) c h w')
                    decoded_intermediate = tokenizer.decode(step_samples_reshaped)
                    decoded_intermediate = rearrange(decoded_intermediate, '(b f) c h w -> b f c h w', b=b)
                    intermediate_results[current_step] = decoded_intermediate

    # Decode the final sampled latents.
    with torch.no_grad():
        b, f, c, h, w = final_samples.shape
        final_samples = rearrange(final_samples, 'b f c h w -> (b f) c h w')
        final_decoded = tokenizer.decode(final_samples)
        final_decoded = rearrange(final_decoded, '(b f) c h w -> b f c h w', b=b)

    if save_intermediate:
        return final_decoded, intermediate_results
    else:
        return final_decoded


def merge_samples_with_cond(generated_samples, video_conditions):
    """
    Interleave generated samples with the original conditioning frames.

    Args:
        generated_samples (torch.Tensor): Generated tensor of shape (F, C, H, W).
        video_conditions (torch.Tensor): Conditioning tensor of shape (F+1, C, H, W).

    Returns:
        torch.Tensor: Merged tensor of shape (F*2+1, C, H, W).
    """
    F, C, H, W = generated_samples.shape

    merged = torch.empty((F * 2 + 1, C, H, W), device=generated_samples.device, dtype=generated_samples.dtype)

    # Interleave: conditioning frames at even indices, generated frames at odd indices
    for i in range(F + 1):
        merged[2 * i] = video_conditions[i]  # conditioning frame

    for i in range(F):
        merged[2 * i + 1] = generated_samples[i]  # generated frame

    return merged


def concatenate_chunks_with_overlap_removal(chunks_list, chunks_info):
    """
    Concatenate chunks and remove overlap from chunks that have it.
    Each chunk (except the first) overlaps with the previous chunk by at least 1 conditioning frame.

    Args:
        chunks_list (list): List of video tensors from each chunk
        chunks_info (list): List of chunk information dictionaries

    Returns:
        torch.Tensor: Concatenated video with overlap removed
    """
    if not chunks_list:
        return torch.empty(0, 3, 512, 512)

    if len(chunks_list) == 1:
        return chunks_list[0]

    result_chunks = []

    for i, (chunk_video, chunk_info) in enumerate(zip(chunks_list, chunks_info)):
        if i == 0:
            # First chunk - no overlap to remove
            result_chunks.append(chunk_video)
        else:
            # All other chunks have overlap with previous chunks
            overlap_frames = chunk_info['overlap_frames']

            if overlap_frames > 0:
                # Remove overlap frames from the beginning of this chunk
                # overlap_frames is the number of frames that overlap with previous chunks
                overlap_to_remove = overlap_frames
                result_chunks.append(chunk_video[overlap_to_remove:])
            else:
                # No overlap, add the entire chunk (this shouldn't happen for i > 0)
                result_chunks.append(chunk_video)

    return torch.cat(result_chunks, dim=0)


def convert_tensor_to_video(merged):
    """
    Convert the merged tensor to a video-friendly uint8 grayscale format.

    Args:
        merged (torch.Tensor): Tensor of shape (F*2+1, C, H, W) with float values.

    Returns:
        torch.Tensor: Converted tensor with shape (F*2+1, H, W) for grayscale video saving/display.
    """
    # Handle empty tensor
    if merged.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)

    # Handle grayscale conversion
    if merged.shape[1] == 3:
        grayscale = merged.mean(dim=1)
    else:
        grayscale = merged.squeeze(1)

    # Convert to uint8 range
    video = ((grayscale * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255)
    video = video.to(dtype=torch.uint8).cpu().contiguous()
    return video


def save_video(video, frames_dir, parent_dir, resize_size=(512, 512), fps=30):
    """
    Save video frames as images in frames_dir and optionally as video in parent_dir.

    Args:
        video (torch.Tensor): Tensor of video frames
        frames_dir (Path): Directory to save individual frames
        parent_dir (Path): Parent directory to save the video file
        resize_size (tuple): Size to resize frames to
        fps (int): Frames per second for video
    """
    # Handle empty video
    if video.numel() == 0:
        print(f"Warning: Empty video tensor, skipping save to {frames_dir}")
        return

    # Save individual frames
    for i, frame in enumerate(video):
        img = Image.fromarray(frame.numpy(), mode='L')  # 'L' mode for grayscale
        img = img.resize(resize_size)
        img.save(os.path.join(frames_dir, f"{i:04d}.png"))

    mp4_path = os.path.join(parent_dir, "video.mp4")
    frames_to_mp4(frames_dir, mp4_path, fps)


def frames_to_mp4(frames_dir, output_path, fps=30):
    """
    Convert a directory of grayscale image frames to an MP4 video file.

    Args:
        frames_dir (str): Directory containing the image frames (numbered sequentially)
        output_path (str): Path to save the output MP4 file
        fps (int): Frames per second for the output video
    """
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]), cv2.IMREAD_COLOR)
    H, W, _ = first_frame.shape

    imgs = []
    for f in frames:
        img = cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_COLOR)
        imgs.append(img)

    # YUV 4:4:4
    iio.imwrite(
        output_path,
        imgs,
        plugin="FFMPEG",
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=[
            "-crf", "10",                   # 18–20 is common; 10 is very high quality
            "-preset", "slow",
            "-pix_fmt", "yuv444p",          # No chroma subsampling
            "-movflags", "+faststart",
        ],
	)

    print(f"Grayscale video saved to {output_path}")


def process_sample_chunks_in_batches(chunks, model, transport, tokenizer, config_args, cfg_scale, batch_size, device, num_sampling_steps,
                                   save_intermediate=False, intermediate_steps=None):
    """
    Process chunks for a single sample in batches.

    Args:
        chunks: List of chunks for a single sample
        model: The main generative model
        transport: Transport process object
        tokenizer: Tokenizer for encoding and decoding
        config_args: Configuration parameters
        cfg_scale: Classifier-free guidance scale
        batch_size: Number of chunks to process in each batch
        device: Torch device
        num_sampling_steps: Number of sampling steps
        save_intermediate: Whether to save intermediate results
        intermediate_steps: Specific steps to save intermediate results

    Returns:
        tuple: (all_generated_clips, all_intermediate_results) where
               all_generated_clips is list of final generated clips and
               all_intermediate_results is list of intermediate results dicts (or None if not saving)
    """
    all_generated_clips = []
    all_intermediate_results = [] if save_intermediate else None

    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:batch_end]
        current_batch_size = len(batch_chunks)

        # Prepare batch data
        batch_cond_frames = torch.stack([chunk['cond_frames'] for chunk in batch_chunks]).to(device)
        if config_args.model.use_ecg:
            batch_ecg_cond = torch.stack([chunk['cond_ecg'] for chunk in batch_chunks]).to(device)
        else:
            batch_ecg_cond = None

        batch_num = i//batch_size + 1
        total_batches = (len(chunks) + batch_size - 1)//batch_size

        # Print progress for this sample's batches
        if total_batches > 1:  # Only print if sample has multiple batches
            print(f"    Processing chunk batch {batch_num}/{total_batches} ({current_batch_size} chunks)")

        # Generate frames for this batch
        if save_intermediate:
            generated_batch, intermediate_batch = generate_sample_for_clip(
                batch_cond_frames, batch_ecg_cond, model, transport, tokenizer,
                config_args, cfg_scale, num_sampling_steps,
                save_intermediate, intermediate_steps
            )
        else:
            generated_batch = generate_sample_for_clip(
                batch_cond_frames, batch_ecg_cond, model, transport, tokenizer,
                config_args, cfg_scale, num_sampling_steps
            )
            intermediate_batch = {}

        # Move to CPU immediately to save GPU memory
        for j in range(current_batch_size):
            all_generated_clips.append(generated_batch[j].cpu())
            if save_intermediate:
                # Store intermediate results for each chunk (move to CPU)
                chunk_intermediate = {}
                for step, step_results in intermediate_batch.items():
                    chunk_intermediate[step] = step_results[j].cpu()
                all_intermediate_results.append(chunk_intermediate)

        # Clear GPU cache after each batch (most important!)
        torch.cuda.empty_cache()

    return all_generated_clips, all_intermediate_results


def concat_images(data_dirs, tags, output_dir, include_original=True):
    video_ids = [i.stem for i in data_dirs[0].glob("*") if i.is_dir()]
    title = " | ".join(tags)
    if include_original:
        title = "original | " + title
    height = 10
    width = height * len(data_dirs)

    print(f"Total {len(video_ids)} videos processing started...")

    for idx, video_id in enumerate(video_ids):
        print(f"[{idx}/{len(video_ids)}] {video_id} processing...")
        frame_paths = list(
            (data_dirs[0] / video_id / "merged" / "frames").glob("*.png")
        )
        frame_names = [x.name for x in frame_paths]
        frame_names = sorted(frame_names, key=lambda x: int(x.split(".")[0]))
        frame_names = [i for i in frame_names if int(i.split(".")[0]) % 2 != 0]

        print(f"  - {len(frame_names)} frames processing...")
        processed_frames = 0

        for fname in frame_names:
            output_subdir = output_dir / video_id
            output_subdir.mkdir(exist_ok=True, parents=True)
            output_fpath = output_subdir / fname
            if output_fpath.exists():
                continue

            frame_imgs = []
            for i, data_dir in enumerate(data_dirs):
                if i == 0 and include_original:
                    original_fpath = data_dir / video_id / "original" / "frames" / fname
                    frame_imgs.append(Image.open(original_fpath))

                fpath = data_dir / video_id / "merged" / "frames" / fname
                img = Image.open(fpath)
                frame_imgs.append(img)

            merged_img = np.concatenate([np.array(img) for img in frame_imgs], axis=1)
            merged_img = Image.fromarray(merged_img)

            plt.figure(figsize=(width, height))
            plt.title(title, fontsize=20)
            plt.imshow(merged_img, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_fpath)
            plt.close("all")
            processed_frames += 1

        print(
            f"  - {video_id} Completed! ({processed_frames}/{len(frame_names)} frames processed)"
        )

    print("All image merging operations completed!")
