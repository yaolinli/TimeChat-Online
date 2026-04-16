"""
Token Drop Visualization for TimeChat-Online (DTD Module)

This script visualizes which visual tokens (patches) are dropped by the
Differential Token Drop (DTD) module. It uses the SAME video processing
pipeline as Qwen2.5-VL (via qwen_vl_utils) to ensure the extracted frames
exactly match what the model sees during inference.

Usage:
    python eval/get_visual_case.py \
        --dp_jsonl eval/sample_151_ft0d4.jsonl \
        --video_path /path/to/video.mp4 \
        --output_dir ./vis_output
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Use the repo's own qwen_vl_utils (same as model inference)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))
from qwen_vl_utils.src.qwen_vl_utils import process_vision_info


def load_drop_positions(dp_jsonl_path):
    """
    Load drop positions from JSONL file.
    
    Format: each line is a JSON list, first element is a dict mapping
    frame_index (str) -> list of [i, j] patch coordinates that were dropped.
    
    Returns:
        dict: {frame_idx(int): [(i, j), ...]}
    """
    with open(dp_jsonl_path, 'r') as f:
        lines = f.readlines()
    
    data = json.loads(lines[-1])
    if isinstance(data, list):
        dp = data[0]
    else:
        dp = data
    
    result = {}
    for key, patches in dp.items():
        result[int(key)] = [(p[0], p[1]) for p in patches]
    
    return result


def extract_frames_qwen(video_path, min_pixels=448*448, max_pixels=448*448,
                         max_frames=1016, min_frames=4, fps=1.0):
    """
    Extract and resize video frames using Qwen2.5-VL's own pipeline.
    This guarantees the frames are IDENTICAL to what the model processes.
    
    Returns:
        list of PIL Images, resized_height, resized_width
    """
    import torch
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "max_frames": max_frames,
                    "min_frames": min_frames,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": "",
                },
            ],
        }
    ]
    
    _, video_inputs = process_vision_info(messages)
    video_tensor = video_inputs[0]  # shape: [T, C, H, W], float
    
    # Convert to uint8 PIL images
    video_tensor = video_tensor.to(torch.uint8)
    T, C, H, W = video_tensor.shape
    
    frames = []
    for t in range(T):
        frame_arr = video_tensor[t].permute(1, 2, 0).numpy()  # [H, W, C]
        frames.append(Image.fromarray(frame_arr))
    
    return frames, H, W


def apply_drop_mask(frame, dropped_patches, patch_size=28, alpha=0.8,
                    color=(255, 255, 255)):
    """
    Apply white overlay on dropped patch regions.
    """
    frame_arr = np.array(frame, dtype=np.float32)
    overlay = np.array(color, dtype=np.float32)
    
    for i, j in dropped_patches:
        h_start = i * patch_size
        h_end = min((i + 1) * patch_size, frame_arr.shape[0])
        w_start = j * patch_size
        w_end = min((j + 1) * patch_size, frame_arr.shape[1])
        
        region = frame_arr[h_start:h_end, w_start:w_end]
        frame_arr[h_start:h_end, w_start:w_end] = region * (1 - alpha) + overlay * alpha
    
    return Image.fromarray(frame_arr.astype(np.uint8))


def make_grid(images, nrow=8, padding=2, bg_color=(0, 0, 0)):
    """Create a grid image from a list of PIL Images."""
    if not images:
        raise ValueError("Empty image list")
    
    w, h = images[0].size
    n = len(images)
    ncol = nrow
    nrows = (n + ncol - 1) // ncol
    
    grid_w = ncol * w + (ncol - 1) * padding
    grid_h = nrows * h + (nrows - 1) * padding
    
    grid = Image.new('RGB', (grid_w, grid_h), bg_color)
    
    for idx, img in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        x = col * (w + padding)
        y = row * (h + padding)
        grid.paste(img, (x, y))
    
    return grid


def visualize_token_drop(dp_jsonl, video_path, output_dir,
                          patch_size=28, fps=1.0, alpha=0.8,
                          grid_nrow=8, max_frames=1016, min_frames=4,
                          min_pixels=448*448, max_pixels=448*448,
                          save_individual=True, save_grid=True):
    """
    Main visualization function.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load drop positions
    print(f"[1/4] Loading drop positions from {dp_jsonl}...")
    drop_positions = load_drop_positions(dp_jsonl)
    print(f"       Found drop data for {len(drop_positions)} frames")
    
    # 2. Extract frames using Qwen2.5-VL's own pipeline
    print(f"[2/4] Extracting frames via Qwen2.5-VL pipeline (fps={fps})...")
    frames, resized_h, resized_w = extract_frames_qwen(
        video_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_frames=max_frames,
        min_frames=min_frames,
        fps=fps,
    )
    
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    print(f"       Extracted {len(frames)} frames, resolution: {resized_h}x{resized_w}")
    print(f"       Patch grid: {grid_h}x{grid_w} = {grid_h * grid_w} patches/frame")
    
    # 3. Apply drop masks
    # DTD temporal_patch_size=2: dp key "t" covers frames 2t and 2t+1
    # Key "0" (first frame pair) is the reference, not dropped
    print(f"[3/4] Applying drop masks (alpha={alpha})...")
    
    raw_frames = []
    masked_frames = []
    stats = []
    
    for frame_idx, frame in enumerate(frames):
        raw_frames.append(frame.copy())
        
        dp_key = frame_idx // 2
        
        if dp_key in drop_positions:
            dropped = drop_positions[dp_key]
            masked = apply_drop_mask(frame, dropped, patch_size=patch_size, alpha=alpha)
            drop_ratio = len(dropped) / (grid_h * grid_w) * 100
        else:
            masked = frame.copy()
            drop_ratio = 0.0
        
        masked_frames.append(masked)
        stats.append((frame_idx, dp_key, drop_ratio))
    
    drop_ratios = [s[2] for s in stats if s[2] > 0]
    avg_drop = sum(drop_ratios) / len(drop_ratios) if drop_ratios else 0
    print(f"       Average drop ratio: {avg_drop:.1f}%")
    
    # 4. Save outputs
    print(f"[4/4] Saving to {output_dir}...")
    
    if save_individual:
        for idx, (raw, masked) in enumerate(zip(raw_frames, masked_frames)):
            raw.save(os.path.join(output_dir, f"{idx:03d}second_raw.png"), dpi=(300, 300))
            masked.save(os.path.join(output_dir, f"{idx:03d}second.png"), dpi=(300, 300))
    
    if save_grid and len(raw_frames) > 0:
        max_grid = grid_nrow * 4
        step = max(1, len(raw_frames) // max_grid)
        grid_raw_imgs = [raw_frames[i] for i in range(0, len(raw_frames), step)][:max_grid]
        grid_masked_imgs = [masked_frames[i] for i in range(0, len(masked_frames), step)][:max_grid]
        
        raw_grid = make_grid(grid_raw_imgs, nrow=grid_nrow)
        masked_grid = make_grid(grid_masked_imgs, nrow=grid_nrow)
        
        raw_grid.save(os.path.join(output_dir, "grid_raw.png"), dpi=(300, 300))
        masked_grid.save(os.path.join(output_dir, "grid_dropped.png"), dpi=(300, 300))
        
        # Side-by-side comparison
        comp_w = raw_grid.width + masked_grid.width + 20
        comp_h = max(raw_grid.height, masked_grid.height) + 40
        comparison = Image.new('RGB', (comp_w, comp_h), (255, 255, 255))
        comparison.paste(raw_grid, (0, 40))
        comparison.paste(masked_grid, (raw_grid.width + 20, 40))
        
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((raw_grid.width // 2 - 60, 10), "Original Frames", fill=(0, 0, 0), font=font)
        draw.text((raw_grid.width + 20 + masked_grid.width // 2 - 100, 10),
                   f"Token Drop ({avg_drop:.1f}% dropped)", fill=(200, 0, 0), font=font)
        
        comparison.save(os.path.join(output_dir, "comparison.png"), dpi=(300, 300))
    
    # Save stats
    stats_path = os.path.join(output_dir, "drop_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            "video": video_path,
            "dp_jsonl": dp_jsonl,
            "total_frames": len(frames),
            "resolution": f"{resized_h}x{resized_w}",
            "patch_grid": f"{grid_h}x{grid_w}",
            "patch_size": patch_size,
            "avg_drop_ratio": f"{avg_drop:.1f}%",
            "per_frame_stats": [
                {"frame": s[0], "dp_key": s[1], "drop_ratio": f"{s[2]:.1f}%"}
                for s in stats
            ]
        }, f, indent=2)
    
    print(f"\nDone! Output saved to {output_dir}/")
    print(f"  - {len(frames)} raw frames: 000second_raw.png ~ {len(frames)-1:03d}second_raw.png")
    print(f"  - {len(frames)} masked frames: 000second.png ~ {len(frames)-1:03d}second.png")
    if save_grid:
        print(f"  - Grid: grid_raw.png, grid_dropped.png, comparison.png")
    print(f"  - Stats: drop_stats.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Token Drop Visualization for TimeChat-Online')
    parser.add_argument('--dp_jsonl', type=str, required=True,
                        help='Path to drop-position JSONL file')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='./vis_output',
                        help='Output directory (default: ./vis_output)')
    parser.add_argument('--patch_size', type=int, default=28,
                        help='Patch size in pixels (default: 28)')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frame extraction rate (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Drop mask opacity, 0-1 (default: 0.8)')
    parser.add_argument('--grid_nrow', type=int, default=8,
                        help='Grid images per row (default: 8)')
    parser.add_argument('--max_frames', type=int, default=1016,
                        help='Max frames to extract (default: 1016)')
    parser.add_argument('--min_frames', type=int, default=4,
                        help='Min frames to extract (default: 4)')
    parser.add_argument('--min_pixels', type=int, default=200704,
                        help='Min pixels per frame (default: 448*448=200704)')
    parser.add_argument('--max_pixels', type=int, default=200704,
                        help='Max pixels per frame (default: 448*448=200704)')
    parser.add_argument('--no_individual', action='store_true',
                        help='Skip saving individual frame images')
    parser.add_argument('--no_grid', action='store_true',
                        help='Skip saving grid images')
    
    args = parser.parse_args()
    
    visualize_token_drop(
        dp_jsonl=args.dp_jsonl,
        video_path=args.video_path,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        fps=args.fps,
        alpha=args.alpha,
        grid_nrow=args.grid_nrow,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        save_individual=not args.no_individual,
        save_grid=not args.no_grid,
    )
