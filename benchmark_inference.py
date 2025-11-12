import argparse
import time
from pathlib import Path
import statistics

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
    parser = argparse.ArgumentParser(description="Benchmark inference time for angio video interpolation")
    parser.add_argument("ckpt_path", type=str, help="Path to checkpoint file")
    parser.add_argument("data_path", type=str, help="Path to test data")
    parser.add_argument("--video_key", type=str, required=True,
                        help="Specific video key to benchmark (e.g., video name)")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Number of iterations to run for benchmarking")
    parser.add_argument("--warmup_iterations", type=int, default=50,
                        help="Number of warmup iterations (not counted in statistics)")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="Number of chunks to process (default: all chunks). Use 1 for single chunk benchmark.")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Output directory for benchmark results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--num_sampling_steps", type=int, default=50,
                        help="Number of sampling steps (ODE for Flow Matching, SDE for Bridge)")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale for sampling")
    parser.add_argument("--use_ecg", action='store_true', help="Use ECG as conditioning")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--save_output", action='store_true',
                        help="Save the output video (only from the last iteration)")

    return parser.parse_args()


def run_single_inference(sample, model, transport, tokenizer, config_args,
                        cfg_scale, batch_size, device, num_sampling_steps, num_chunks=None):
    """Run inference on a single sample and return the processing time."""
    chunks = sample['chunks']

    # Limit chunks if specified
    if num_chunks is not None and num_chunks > 0:
        chunks = chunks[:num_chunks]

    # Start timing
    start_time = time.time()

    # Process chunks for this sample in batches
    all_generated_clips, _ = process_sample_chunks_in_batches(
        chunks, model, transport, tokenizer,
        config_args, cfg_scale, batch_size, device, num_sampling_steps,
        save_intermediate=False, intermediate_steps=[]
    )

    # Organize results
    all_pred_clips = []

    # Process each chunk result
    for chunk_idx, chunk in enumerate(chunks):
        generated_clip_for_chunk = all_generated_clips[chunk_idx]
        pred_frames_this_chunk = generated_clip_for_chunk.to(device)
        cond_frames_this_chunk = chunk['cond_frames'].to(device)
        merged_pred_video_for_chunk = merge_samples_with_cond(pred_frames_this_chunk, cond_frames_this_chunk)
        all_pred_clips.append(merged_pred_video_for_chunk)

    # Concatenate all clips with overlap removal
    pred_video = concatenate_chunks_with_overlap_removal(all_pred_clips, chunks)

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time

    return inference_time, pred_video


def main():
    # Parse command line arguments
    args = parse_args()

    # Extract specific arguments
    ckpt_path = Path(args.ckpt_path)
    result_dir = ckpt_path.parent.parent
    config_path = result_dir / "config.yaml"
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    device = args.device
    num_sampling_steps = args.num_sampling_steps
    use_ecg = args.use_ecg
    batch_size = args.batch_size
    cfg_scale = args.cfg_scale
    video_key = args.video_key
    num_iterations = args.num_iterations
    warmup_iterations = args.warmup_iterations
    num_chunks = args.num_chunks

    # Load config file settings
    config_args = OmegaConf.load(config_path)
    config_args.model.use_ecg = use_ecg

    # Display sampling method info
    if config_args.transport.path_type == "Bridge":
        print(f"Bridge path detected. Will use SDE Euler sampling with constant diffusion.")
    else:
        print(f"{config_args.transport.path_type} path detected. Will use ODE Euler sampling.")

    print(f"\n{'='*80}")
    print(f"Inference Benchmark Configuration")
    print(f"{'='*80}")
    print(f"Video key: {video_key}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Chunks to process: {num_chunks if num_chunks else 'All'}")
    print(f"Sampling steps: {num_sampling_steps}")
    print(f"CFG scale: {cfg_scale}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    print("Loading models...")
    model, transport, tokenizer = setup_models(config_args, ckpt_path, device)
    print("Models loaded successfully.\n")

    # Create the dataset
    dataset = create_alternating_dataset(
        data_path,
        image_size=config_args.model.image_size,
        num_frames=config_args.model.num_frames,
        use_ecg=config_args.model.use_ecg,
        ecg_signal_len=config_args.model.ecg_signal_len,
        test_mode=True
    )
    torch.set_grad_enabled(False)

    # Find the specific video sample
    target_sample = None
    for sample in dataset:
        if sample['key'] == video_key:
            target_sample = sample
            break

    if target_sample is None:
        print(f"Error: Video key '{video_key}' not found in dataset.")
        print(f"Available video keys:")
        for sample in dataset:
            print(f"  - {sample['key']}")
        return

    chunks = target_sample['chunks']
    if not chunks:
        print(f"Error: Video '{video_key}' has no chunks (might be too short).")
        return

    total_chunks_available = len(chunks)
    chunks_to_use = num_chunks if num_chunks and num_chunks < total_chunks_available else total_chunks_available

    print(f"Found video: {video_key}")
    print(f"Total chunks available: {total_chunks_available}")
    print(f"Chunks to process: {chunks_to_use}\n")

    # Warmup iterations
    if warmup_iterations > 0:
        print(f"Running {warmup_iterations} warmup iteration(s)...")
        for i in range(warmup_iterations):
            warmup_time, _ = run_single_inference(
                target_sample, model, transport, tokenizer,
                config_args, cfg_scale, batch_size, device, num_sampling_steps, num_chunks
            )
            torch.cuda.empty_cache()
            print(f"  Warmup {i+1}/{warmup_iterations}: {warmup_time:.3f}s")
        print("Warmup completed.\n")

    # Benchmark iterations
    print(f"Running {num_iterations} benchmark iteration(s)...")
    inference_times = []
    last_pred_video = None

    for i in range(num_iterations):
        iteration_time, pred_video = run_single_inference(
            target_sample, model, transport, tokenizer,
            config_args, cfg_scale, batch_size, device, num_sampling_steps, num_chunks
        )
        inference_times.append(iteration_time)
        last_pred_video = pred_video

        torch.cuda.empty_cache()
        print(f"  Iteration {i+1}/{num_iterations}: {iteration_time:.3f}s")

    print("\nBenchmark completed.\n")

    # Calculate statistics
    mean_time = statistics.mean(inference_times)
    median_time = statistics.median(inference_times)
    stdev_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
    min_time = min(inference_times)
    max_time = max(inference_times)

    # Print results
    print(f"{'='*80}")
    print(f"Inference Benchmark Results")
    print(f"{'='*80}")
    print(f"Video: {video_key}")
    print(f"Number of chunks processed: {chunks_to_use}")
    if num_chunks and num_chunks < total_chunks_available:
        print(f"  (Total chunks available: {total_chunks_available})")
    print(f"Total iterations: {num_iterations}")
    print(f"\nTiming Statistics:")
    print(f"  Mean:     {mean_time:.3f}s ± {stdev_time:.3f}s")
    print(f"  Median:   {median_time:.3f}s")
    print(f"  Min:      {min_time:.3f}s")
    print(f"  Max:      {max_time:.3f}s")
    print(f"  Range:    {max_time - min_time:.3f}s")

    if stdev_time > 0:
        cv = (stdev_time / mean_time) * 100
        print(f"  CV:       {cv:.2f}%")

    print(f"\nPer-chunk statistics:")
    print(f"  Mean time per chunk: {mean_time / chunks_to_use:.3f}s")
    print(f"{'='*80}\n")

    # Save results to file
    results_file = output_dir / f"benchmark_{video_key}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Inference Benchmark Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Video: {video_key}\n")
        f.write(f"Number of chunks processed: {chunks_to_use}\n")
        if num_chunks and num_chunks < total_chunks_available:
            f.write(f"  (Total chunks available: {total_chunks_available})\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Sampling steps: {num_sampling_steps}\n")
        f.write(f"  CFG scale: {cfg_scale}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Warmup iterations: {warmup_iterations}\n")
        f.write(f"  Benchmark iterations: {num_iterations}\n")
        f.write(f"\nTiming Statistics:\n")
        f.write(f"  Mean:     {mean_time:.3f}s ± {stdev_time:.3f}s\n")
        f.write(f"  Median:   {median_time:.3f}s\n")
        f.write(f"  Min:      {min_time:.3f}s\n")
        f.write(f"  Max:      {max_time:.3f}s\n")
        f.write(f"  Range:    {max_time - min_time:.3f}s\n")
        if stdev_time > 0:
            cv = (stdev_time / mean_time) * 100
            f.write(f"  CV:       {cv:.2f}%\n")
        f.write(f"\nPer-chunk statistics:\n")
        f.write(f"  Mean time per chunk: {mean_time / chunks_to_use:.3f}s\n")
        f.write(f"\nIndividual iteration times:\n")
        for i, t in enumerate(inference_times, 1):
            f.write(f"  Iteration {i}: {t:.3f}s\n")

    print(f"Results saved to: {results_file}")

    # Optionally save the output video from the last iteration
    if args.save_output and last_pred_video is not None:
        print(f"\nSaving output video from last iteration...")
        video_output_dir = output_dir / video_key
        video_output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = video_output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        video_to_save = convert_tensor_to_video(last_pred_video)
        save_video(video_to_save, frames_dir, video_output_dir, fps=15)
        print(f"Output saved to: {video_output_dir}")


if __name__ == '__main__':
    main()
