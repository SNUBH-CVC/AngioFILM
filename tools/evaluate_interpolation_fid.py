#!/usr/bin/env python3
"""
Evaluate interpolation results using FID (Fréchet Inception Distance) metric.

This script is specifically designed for interpolation results with the structure:
outputs/experiment_name/
├── video_key1/
│   ├── original/frames/*.png
│   ├── merged/frames/*.png
│   └── comparison/frames/*.png

Usage:
    python tools/evaluate_interpolation_fid.py ./outputs/experiment_name
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pyiqa

SAVE_FILENAME = "fid_results.json"


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


def collect_images_for_fid(experiments: List[Tuple[str, Path]], temp_ref_dir: Path,
                          temp_dist_dir: Path, include_original: bool) -> Tuple[int, int]:
    """
    Collect reference and distribution images for FID calculation.

    Returns:
        Tuple of (ref_count, dist_count)
    """
    ref_count = 0
    dist_count = 0

    for video_key, video_dir in experiments:
        # Collect reference images (original)
        original_frames_dir = video_dir / "original" / "frames"
        if original_frames_dir.exists():
            ref_imgs = sorted(list(original_frames_dir.glob("*.png")))

            # Filter frames based on include_original option
            if include_original:
                # Use all frames
                frames_to_use = ref_imgs
            else:
                # Use only odd indices (interpolated frames)
                frames_to_use = [f for i, f in enumerate(ref_imgs) if i % 2 == 1]

            for img_path in frames_to_use:
                new_name = f"{video_key}_{img_path.stem}_original.png"
                shutil.copyfile(img_path, temp_ref_dir / new_name)
                ref_count += 1

        # Collect distribution images (merged)
        comp_frames_dir = video_dir / "merged" / "frames"
        if comp_frames_dir.exists():
            dist_imgs = sorted(list(comp_frames_dir.glob("*.png")))

            # Filter frames based on include_original option
            if include_original:
                # Use all frames
                frames_to_use = dist_imgs
            else:
                # Use only odd indices (interpolated frames)
                frames_to_use = [f for i, f in enumerate(dist_imgs) if i % 2 == 1]

            for img_path in frames_to_use:
                new_name = f"{video_key}_{img_path.stem}_merged.png"
                shutil.copyfile(img_path, temp_dist_dir / new_name)
                dist_count += 1

        import glob
        import os
        import sys
        temp_ref_imgs = sorted(glob.glob(os.path.join(temp_ref_dir, "*.png")))
        temp_dist_imgs = sorted(glob.glob(os.path.join(temp_dist_dir, "*.png")))
        for ref_p, dist_p in zip(temp_ref_imgs, temp_dist_imgs):
            print(ref_p)
            print(dist_p)
            print("\n")
        sys.exit()
    return ref_count, dist_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate interpolation results using FID metric")
    parser.add_argument("output_dir", type=str, help="Directory containing interpolation results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for FID calculation")
    parser.add_argument("--include_original", action="store_true",
                        help="Include original frames in evaluation (evaluate all frames). "
                             "If not set, only evaluate interpolated frames (odd indices).")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for FID calculation (default: output_dir/temp_fid)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    print(f"Evaluating interpolation FID in: {output_dir}")

    # Set up temporary directories
    if args.temp_dir:
        temp_base_dir = Path(args.temp_dir)
    else:
        temp_base_dir = output_dir / "temp_fid"

    temp_ref_dir = temp_base_dir / "reference"
    temp_dist_dir = temp_base_dir / "distribution"

    # Find video experiments
    experiments = find_video_experiments(output_dir)

    if not experiments:
        raise FileNotFoundError(f"No interpolation experiments found in {output_dir}. "
                               f"Expected structure: video_key/original/frames/ and video_key/merged/frames/")

    print(f"Found {len(experiments)} video experiments to evaluate")
    if args.include_original:
        print("Mode: Evaluating all frames (including original frames)")
    else:
        print("Mode: Evaluating only interpolated frames (odd indices)")

    # Create FID metric
    print(f"Initializing FID metric on {args.device}...")
    fid_dinov2_metric = pyiqa.create_metric("fid_dinov2", device=args.device)

    print("FID metric initialized successfully")

    print(f"\n=== Processing FID evaluation ===")

    # Clean and create temporary directories
    if temp_ref_dir.exists():
        shutil.rmtree(temp_ref_dir)
    if temp_dist_dir.exists():
        shutil.rmtree(temp_dist_dir)

    temp_ref_dir.mkdir(parents=True, exist_ok=True)
    temp_dist_dir.mkdir(parents=True, exist_ok=True)

    # Check if all experiments have merged frames
    valid_experiments = []
    for video_key, video_dir in experiments:
        comp_frames_dir = video_dir / "merged" / "frames"
        if comp_frames_dir.exists() and any(comp_frames_dir.glob("*.png")):
            valid_experiments.append((video_key, video_dir))

    if not valid_experiments:
        raise ValueError("No valid experiments found with merged frames")

    print(f"  Found {len(valid_experiments)} valid experiments")

    # Collect images for FID calculation
    print(f"  Collecting images for FID calculation...")
    ref_count, dist_count = collect_images_for_fid(
        valid_experiments, temp_ref_dir, temp_dist_dir, args.include_original
    )

    if ref_count == 0 or dist_count == 0:
        raise ValueError(f"Insufficient images for FID calculation: {ref_count} ref, {dist_count} dist")

    print(f"  Collected {ref_count} reference images and {dist_count} distribution images")

    # Calculate FID
    print(f"  Calculating FID...")
    fid_dinov2_score = fid_dinov2_metric(str(temp_dist_dir), str(temp_ref_dir), clean=True)
    fid_dinov2_value = float(fid_dinov2_score)
    print(f"  FID Dinov2 Score: {fid_dinov2_value:.4f}")

    # Prepare successful result
    comparison_type = "merged"
    result = {
        'comparison_type': comparison_type,
        'fid_dinov2_score': fid_dinov2_value,
        'num_ref_imgs': ref_count,
        'num_dist_imgs': dist_count,
        'num_videos': len(valid_experiments),
        'video_keys': ';'.join([vk for vk, _ in valid_experiments]),
        'include_original': args.include_original,
        'success': True
    }

    # Prepare JSON output structure (only for successful cases)
    output_data = {
        "config": {
            "output_dir": str(output_dir.absolute()),
            "device": args.device,
            "include_original": args.include_original,
            "temp_dir": str(temp_base_dir.absolute()) if args.temp_dir else None,
            "timestamp": datetime.now().isoformat()
        },
        "results": {
            "fid_dinov2_score": result['fid_dinov2_score'],
            "comparison_type": result['comparison_type'],
            "success": True
        },
        "metadata": {
            "num_ref_imgs": result['num_ref_imgs'],
            "num_dist_imgs": result['num_dist_imgs'],
            "num_videos": result['num_videos'],
            "video_keys": result['video_keys'].split(';') if result['video_keys'] else []
        }
    }

    # Save results to JSON
    results_file = output_dir / SAVE_FILENAME
    print(f"\nSaving results to {results_file}")

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Clean up temporary directories
    if temp_ref_dir.exists():
        shutil.rmtree(temp_ref_dir)
    if temp_dist_dir.exists():
        shutil.rmtree(temp_dist_dir)
    if temp_base_dir.exists() and not any(temp_base_dir.iterdir()):
        temp_base_dir.rmdir()

    print(f"\nResults saved to: {results_file.absolute()}")
    print(f"Evaluation completed!")


if __name__ == "__main__":
    main()
