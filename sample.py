import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dataset import create_alternating_dataset
from inference_utils import (concatenate_chunks_with_overlap_removal,
                             convert_tensor_to_video, merge_samples_with_cond,
                             process_sample_chunks_in_batches, save_video, setup_models)

# Configure torch backends for performance.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Angio video interpolation with flow matching")
    parser.add_argument("ckpt_path", type=str, help="Path to checkpoint file")
    parser.add_argument("data_path", type=str, help="Path to test data")
    parser.add_argument("--output_root_dir", type=str, default="./outputs",
                        help="Output directory for generated samples (defaults to ./outputs/{result_dir.stem})")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--num_sampling_steps", type=int, default=50,
                        help="Number of sampling steps (ODE for Flow Matching, SDE for Bridge)")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale for sampling")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video")
    parser.add_argument("--use_ecg", action='store_true', help="Use ECG as conditioning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--save_mp4", action='store_true', help="Save videos as MP4 files")
    # Intermediate results saving options
    parser.add_argument("--save_intermediate", action='store_true',
                        help="Save intermediate results during sampling")
    parser.add_argument("--intermediate_steps", type=str, default="",
                        help="Specific steps to save (comma-separated, e.g., '10,25,40')")

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Extract specific arguments
    ckpt_path = Path(args.ckpt_path)
    result_dir = ckpt_path.parent.parent
    config_path = result_dir / "config.yaml"
    data_path = Path(args.data_path)
    output_root_dir = Path(args.output_root_dir)
    output_dir = output_root_dir / result_dir.stem
    device = args.device
    num_sampling_steps = args.num_sampling_steps
    use_ecg = args.use_ecg
    batch_size = args.batch_size
    cfg_scale = args.cfg_scale

    # Intermediate results saving options
    save_intermediate = args.save_intermediate
    intermediate_steps = []
    if args.intermediate_steps:
        intermediate_steps = [int(x.strip()) for x in args.intermediate_steps.split(',') if x.strip()]

    # Load config file settings
    config_args = OmegaConf.load(config_path)

    # Override config with CLI settings
    config_args.model.use_ecg = use_ecg

    # Display sampling method info
    if config_args.transport.path_type == "Bridge":
        print(f"Bridge path detected. Will use SDE Euler sampling with constant diffusion.")
    else:
        print(f"{config_args.transport.path_type} path detected. Will use ODE Euler sampling.")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models.
    model, transport, tokenizer = setup_models(config_args, ckpt_path, device)

    # Create the dataset and retrieve one sample.
    dataset = create_alternating_dataset(
        data_path,
        image_size=config_args.model.image_size,
        num_frames=config_args.model.num_frames,
        use_ecg=config_args.model.use_ecg,
        ecg_signal_len=config_args.model.ecg_signal_len,
        test_mode=True
    )
    torch.set_grad_enabled(False)

    # Process each sample individually
    print("Processing samples individually with batch processing per sample...")
    sample_count = 0
    processed_count = 0
    skipped_count = 0
    total_samples = len(dataset)

    # Start total processing time measurement
    total_start_time = time.time()
    inference_times = []

    for sample in dataset:
        sample_count += 1
        video_key = sample['key']
        pred_output_dir = output_dir / video_key

        # Skip if we've already processed this video
        if pred_output_dir.exists():
            print(f"[{sample_count}/{total_samples}] Skipping {video_key}: Output directory already exists")
            skipped_count += 1
            continue

        chunks = sample['chunks']
        if not chunks:
            print(f"[{sample_count}/{total_samples}] Skipping {video_key}: Video too short to create any chunks")
            skipped_count += 1
            continue

        processed_count += 1
        sample_chunk_count = len(chunks)
        print(f"[{processed_count}/{total_samples}] Processing sample: {video_key} ({sample_chunk_count} chunks)")

        # Start sample processing time measurement
        sample_start_time = time.time()

        # Process chunks for this sample in batches
        all_generated_clips, all_intermediate_results = process_sample_chunks_in_batches(
            chunks, model, transport, tokenizer,
            config_args, cfg_scale, batch_size, device, num_sampling_steps,
            save_intermediate, intermediate_steps
        )

        # Organize results for this sample
        all_pred_clips = []
        all_target_clips = []

        # Process each chunk result for this sample
        for chunk_idx, chunk in enumerate(chunks):
            # Get the generated result for this chunk
            generated_clip_for_chunk = all_generated_clips[chunk_idx]  # Shape (F, C, H, W)

            # Prepare frames for merging for this specific chunk
            pred_frames_this_chunk = generated_clip_for_chunk.to(device)  # Shape (F, C, H, W)
            target_frames_this_chunk = chunk['frames'].to(device)  # Shape (F, C, H, W)
            cond_frames_this_chunk = chunk['cond_frames'].to(device)  # Shape (F+1, C, H, W)

            # Merge generated and condition frames for the current chunk
            merged_pred_video_for_chunk = merge_samples_with_cond(pred_frames_this_chunk, cond_frames_this_chunk)
            merged_target_video_for_chunk = merge_samples_with_cond(target_frames_this_chunk, cond_frames_this_chunk)

            all_pred_clips.append(merged_pred_video_for_chunk)  # (2F+1, C, H, W)
            all_target_clips.append(merged_target_video_for_chunk)  # (2F+1, C, H, W)

        # Concatenate all clips with overlap removal
        pred_video = concatenate_chunks_with_overlap_removal(all_pred_clips, chunks)
        target_video = concatenate_chunks_with_overlap_removal(all_target_clips, chunks)

        # Process and save intermediate results for the entire video
        if save_intermediate and all_intermediate_results:
            print(f"    Processing intermediate results for entire video...")

            # Collect all steps that have intermediate results
            all_steps = set()
            for chunk_intermediate in all_intermediate_results:
                all_steps.update(chunk_intermediate.keys())

            # For each step, concatenate all chunks and save
            for step_num in sorted(all_steps):
                step_chunks = []
                step_chunks_info = []

                # Collect intermediate results for this step from all chunks
                for chunk_idx, chunk_intermediate in enumerate(all_intermediate_results):
                    if step_num in chunk_intermediate:
                        # Get the intermediate result for this chunk at this step
                        step_generated_frames = chunk_intermediate[step_num]  # Shape (F, C, H, W)
                        chunk = chunks[chunk_idx]
                        cond_frames = chunk['cond_frames'].cpu()  # Shape (F+1, C, H, W)

                        # Merge with conditioning frames for this chunk
                        merged_step_video = merge_samples_with_cond(step_generated_frames, cond_frames)
                        step_chunks.append(merged_step_video)
                        step_chunks_info.append(chunk)

                # Concatenate all chunks for this step
                if step_chunks:
                    step_full_video = concatenate_chunks_with_overlap_removal(step_chunks, step_chunks_info)

                    # Convert to video-friendly format
                    step_video_to_save = convert_tensor_to_video(step_full_video)

                    # Create step-specific directory
                    step_dir = pred_output_dir / "intermediate" / f"step_{step_num:03d}"
                    step_frames_dir = step_dir / "frames"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    step_frames_dir.mkdir(parents=True, exist_ok=True)

                    # Save frames and video for this step
                    save_video(step_video_to_save, step_frames_dir, step_dir, fps=args.fps)
                    print(f"      Step {step_num}: saved full video to {step_dir}")

        # Convert tensors to video-friendly format
        video_target_to_save = convert_tensor_to_video(target_video)
        video_pred_to_save = convert_tensor_to_video(pred_video)

        # Save original frames and video
        original_dir = pred_output_dir / "original"
        merged_dir = pred_output_dir / "merged"
        original_frames_dir = original_dir / "frames"
        merged_frames_dir = merged_dir / "frames"
        original_dir.mkdir(parents=True, exist_ok=True)
        merged_dir.mkdir(parents=True, exist_ok=True)
        original_frames_dir.mkdir(parents=True, exist_ok=True)
        merged_frames_dir.mkdir(parents=True, exist_ok=True)

        save_video(video_target_to_save, original_frames_dir, original_dir, fps=args.fps)
        save_video(video_pred_to_save, merged_frames_dir, merged_dir, fps=args.fps)

        # Create comparison frames and video by concatenating original and merged tensors in memory
        if video_target_to_save.numel() > 0 and video_target_to_save.shape == video_pred_to_save.shape:
            comparison_dir = pred_output_dir / "comparison"
            comparison_frames_dir = comparison_dir / "frames"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            comparison_frames_dir.mkdir(parents=True, exist_ok=True)

            # Concatenate along the width dimension (dim=2 for tensors with shape [F, H, W])
            comparison_video_tensor = torch.cat((video_target_to_save, video_pred_to_save), dim=2)

            # The default resize in save_video is (512, 512), which is (width, height) for PIL.
            # Adjust the resize for the concatenated video.
            resize_w, resize_h = 512, 512
            comparison_resize_size = (resize_w * 2, resize_h)

            save_video(
                comparison_video_tensor,
                comparison_frames_dir,
                comparison_dir,
                resize_size=comparison_resize_size,
                fps=args.fps
            )
            if args.save_mp4:
                print(f"Comparison video saved to {comparison_dir / 'video.mp4'}")

        # Calculate and record sample processing time
        sample_end_time = time.time()
        sample_inference_time = sample_end_time - sample_start_time
        inference_times.append(sample_inference_time)

        print(f"[{processed_count}/{total_samples}] Completed: {video_key} (Time: {sample_inference_time:.2f}s)")

        # Clear GPU memory after each sample (essential!)
        torch.cuda.empty_cache()

    # Calculate total processing time and statistics
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    print("\nProcessing Summary:")
    print(f"  Total samples in dataset: {sample_count}")
    print(f"  Skipped samples: {skipped_count}")
    print(f"  Successfully processed samples: {processed_count}")
    print(f"  Total processing time: {total_processing_time:.2f}s")

    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        print(f"  Average inference time per sample: {avg_inference_time:.2f}s")
        print(f"  Min inference time: {min_inference_time:.2f}s")
        print(f"  Max inference time: {max_inference_time:.2f}s")


if __name__ == '__main__':
    main()
