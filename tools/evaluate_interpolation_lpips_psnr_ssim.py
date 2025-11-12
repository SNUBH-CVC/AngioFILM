"""
Evaluate interpolation results using LPIPS, PSNR, and SSIM metrics.

This script is specifically designed for interpolation results with the structure:
outputs/experiment_name/
├── video_key1/
│   ├── original/frames/*.png
│   ├── merged/frames/*.png
│   └── comparison/frames/*.png

Usage:
    python tools/evaluate_interpolation_lpips_psnr_ssim.py ./outputs/experiment_name
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyiqa
from PIL import Image

CSV_FILENAME = "lpips_psnr_ssim_results.csv"
JSON_FILENAME = "lpips_psnr_ssim_results.json"


def calculate_all_metrics(original_img: Image.Image, reconstructed_img: Image.Image, metrics_dict: Dict) -> Dict:
    """Calculate all metrics for a pair of images"""
    scores = {}
    for metric_name, metric_func in metrics_dict.items():
        try:
            score = metric_func(original_img, reconstructed_img)
            # Convert tensor to float if needed
            if hasattr(score, 'item'):
                score = score.item()
            scores[f'{metric_name}_score'] = score
        except Exception:
            scores[f'{metric_name}_score'] = None
    return scores


def find_video_experiments(output_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find video experiment directories.
    Returns list of (video_key, video_dir) tuples.
    """
    experiments = []

    for video_dir in output_dir.iterdir():
        if not video_dir.is_dir():
            continue

        # Check for interpolation structure: video_key/original/frames/ and video_key/merged/frames/
        original_frames_dir = video_dir / "original" / "frames"
        merged_frames_dir = video_dir / "merged" / "frames"

        if original_frames_dir.exists() and merged_frames_dir.exists():
            experiments.append((video_dir.name, video_dir))

    return experiments


def is_video_fully_processed(video_dir: Path) -> bool:
    """
    Check if a video has been fully processed by checking if results CSV exists.
    Since CSV is only saved at the very end of processing, its existence means completion.
    """
    results_csv_path = video_dir / CSV_FILENAME
    return results_csv_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Evaluate interpolation results using LPIPS, PSNR, and SSIM")
    parser.add_argument("output_dir", type=str, help="Directory containing interpolation results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for metric calculation")
    parser.add_argument("--include_original", action="store_true",
                        help="Include original frames in evaluation (evaluate all frames). "
                             "If not set, only evaluate interpolated frames (odd indices).")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    # Initialize metrics
    device = args.device
    metric_names = ["lpips", "psnr", "ssim"]
    comparison_type = "merged"  # Fixed to merged since it's always merged

    metrics_dict = {}
    for metric_name in metric_names:
        try:
            metrics_dict[metric_name] = pyiqa.create_metric(metric_name, device=device)
        except Exception as e:
            print(f"Failed to initialize {metric_name}: {e}")
            continue

    if not metrics_dict:
        print("No metrics could be initialized. Exiting.")
        return

    print(f"Initialized metrics: {list(metrics_dict.keys())}")

    # Track errors during processing
    processing_errors = []

    # Find video experiments
    experiments = find_video_experiments(output_dir)

    if not experiments:
        print(f"No interpolation experiments found in {output_dir}")
        print("Expected structure: video_key/original/frames/ and video_key/merged/frames/")
        return

    mode = "all frames" if args.include_original else "interpolated frames only"
    print(f"Found {len(experiments)} experiments, mode: {mode}")

    total_processed = 0

    # Process each video experiment
    for video_key, video_dir in experiments:
        # Check if video is already fully processed
        if is_video_fully_processed(video_dir):
            print(f"Skipping {video_key} (already processed)")
            continue

        print(f"Processing {video_key}...")

        try:
            # Create results CSV path for this video
            video_results_csv_path = video_dir / CSV_FILENAME

            original_frames_dir = video_dir / "original" / "frames"

            # Get list of original images
            original_image_files = sorted(list(original_frames_dir.glob("*.png")))
            if not original_image_files:
                raise FileNotFoundError(f"No PNG files found in {original_frames_dir}")

            # Filter frames based on include_original option
            if args.include_original:
                # Evaluate all frames
                frames_to_evaluate = original_image_files
            else:
                # Evaluate only odd indices (interpolated frames)
                frames_to_evaluate = [f for i, f in enumerate(original_image_files) if i % 2 == 1]

            comparison_frames_dir = video_dir / comparison_type / "frames"
            if not comparison_frames_dir.exists():
                raise FileNotFoundError(f"Required directory not found: {comparison_frames_dir}")

            frames_processed = 0
            results_list = []  # Collect all results in a list

            # Process each frame
            for original_img_path in frames_to_evaluate:
                image_filename = original_img_path.name

                # Check if comparison image exists
                comparison_img_path = comparison_frames_dir / image_filename
                if not comparison_img_path.exists():
                    raise FileNotFoundError(f"Required image not found: {comparison_img_path}")

                # Load images
                original_img = Image.open(original_img_path)
                comparison_img = Image.open(comparison_img_path)

                # Calculate metrics
                metric_scores = calculate_all_metrics(original_img, comparison_img, metrics_dict)

                # Create new result row
                new_row = {
                    'video_key': video_key,
                    'image_name': image_filename,
                    'comparison_type': comparison_type,
                    **metric_scores
                }

                # Add to results list (no concat needed)
                results_list.append(new_row)

                frames_processed += 1
                total_processed += 1

                # Show progress every 20 frames
                if frames_processed % 20 == 0:
                    print(f"  Progress: {frames_processed}/{len(frames_to_evaluate)} frames")

            # Create DataFrame from all results at once (efficient and no warnings)
            video_results_df = pd.DataFrame(results_list, columns=[
                'video_key', 'image_name', 'comparison_type',
                'lpips_score', 'psnr_score', 'ssim_score'
            ])

            # Save after each video (only at the very end)
            video_results_df.to_csv(video_results_csv_path, index=False)
            print(f"  Completed: {frames_processed} frames")

        except Exception as e:
            error_info = {
                "video_key": video_key,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            processing_errors.append(error_info)
            print(f"Failed to process {video_key}: {str(e)}")
            # CSV will not be saved, so this video will be retried next time
            continue

    # Generate summary statistics from individual results
    print("\nGenerating summary statistics...")
    all_results = []
    for video_key, video_dir in experiments:
        video_results_csv_path = video_dir / CSV_FILENAME
        if video_results_csv_path.exists():
            video_df = pd.read_csv(video_results_csv_path)
            all_results.append(video_df)

    summary_stats = []
    if all_results:
        # Only for summary statistics, don't save combined file
        combined_df = pd.concat(all_results, ignore_index=True)

        comp_df = combined_df[combined_df['comparison_type'] == comparison_type]
        if len(comp_df) > 0:
            stats_row = {
                'comparison_type': comparison_type,
                'total_evaluations': len(comp_df),
                'unique_videos': comp_df['video_key'].nunique(),
                'include_original': args.include_original
            }

            for metric in ['lpips_score', 'psnr_score', 'ssim_score']:
                if metric in comp_df.columns:
                    valid_scores = comp_df[metric].dropna()
                    if len(valid_scores) > 0:
                        mean_val = valid_scores.mean()
                        std_val = valid_scores.std()
                        min_val = valid_scores.min()
                        max_val = valid_scores.max()
                        stats_row[f'{metric}_mean'] = mean_val
                        stats_row[f'{metric}_std'] = std_val
                        stats_row[f'{metric}_min'] = min_val
                        stats_row[f'{metric}_max'] = max_val
                        stats_row[f'{metric}_count'] = len(valid_scores)
                    else:
                        stats_row[f'{metric}_mean'] = None
                        stats_row[f'{metric}_std'] = None
                        stats_row[f'{metric}_min'] = None
                        stats_row[f'{metric}_max'] = None
                        stats_row[f'{metric}_count'] = 0

            summary_stats.append(stats_row)

    # Save comprehensive results with summary and errors in JSON
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "total_videos_processed": len([d for _, d in experiments if is_video_fully_processed(d)]),
        "total_videos_found": len(experiments),
        "total_errors": len(processing_errors),
        "include_original": args.include_original,
        "comparison_type": comparison_type,
        "summary_statistics": summary_stats,
        "processing_errors": processing_errors
    }

    json_file = output_dir / JSON_FILENAME
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    if processing_errors:
        print(f"Results saved: {json_file} (with {len(processing_errors)} errors)")
    else:
        print(f"Results saved: {json_file}")

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
