import pandas as pd
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
from qwen_vl_utils_custom import process_vision_info
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from moviepy.editor import VideoFileClip
import numpy as np
import sys
sys.path.append('/home/gaohuan03/yaolinli/code/tokendrop_eval')

def make_custom_grid(imgs, nrow=4, pad=2, big_pad=20):
    """
    参数：
        imgs: list[Tensor]，每个 Tensor 的形状为 (C, H, W)，数据类型为 torch.uint8
        nrow: 每行显示的图片数
        pad: 图片之间的普通间隔宽度（像素）
        big_pad: 每两行后较大间隔的高度（像素）
    返回：
        grid: 拼接后的图片 Tensor，背景使用 0 填充
    """
    N = len(imgs)
    if N == 0:
        raise ValueError
    C, H, W = imgs[0].shape

    n_rows = (N + nrow - 1) // nrow  # 计算行数
    rows = []
    for r in range(n_rows):
        row_imgs = []
        # 对每一行收集 nrow 张图片，不足的补充背景图片
        for c in range(nrow):
            idx = r * nrow + c
            if idx < N:
                row_imgs.append(imgs[idx])
            else:
                row_imgs.append(torch.full((C, H, W), 0, dtype=torch.uint8))
        # 构造横向间隔
        gap_tensor = torch.full((C, H, pad), 0, dtype=torch.uint8)
        row = row_imgs[0]
        for im in row_imgs[1:]:
            row = torch.cat([row, gap_tensor, im], dim=2)
        rows.append(row)
        # 如果不是最后一行，则在行之间插入垂直间隔
        if r < n_rows - 1:
            # 每两行后使用较大的间隔
            gap_h = big_pad if (r + 1) % 2 == 0 else pad
            v_gap = torch.full((C, gap_h, row.shape[2]), 0, dtype=torch.uint8)
            rows.append(v_gap)
    # 将所有行（含间隔）竖向拼接
    grid = rows[0]
    for r in rows[1:]:
        grid = torch.cat([grid, r], dim=1)
    return grid


def get_visual_case(dp_jsonl, video_path, ngrid=32, nrow=8):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 448*448,
                    "max_pixels": 448*448,
                    "max_frames": 1016,
                    "min_frames": 4,
                    "fps": 1,
                    # "resized_height": 448,
                    # "resized_width": 448,
                },
                {
                    "type": "text", 
                    "text": "",
                },
            ],
        }
    ]
    _, video_inputs = process_vision_info(messages)
    video_tensor = video_inputs[0].to(torch.uint8)
    # frame_list = [video_tensor[i] for i in range(video_tensor.shape[0])][:ngrid*2]
    frame_list = [video_tensor[i] for i in range(video_tensor.shape[0])]
    
    with open(dp_jsonl, 'r') as f:
        lines = f.readlines()
    dp = json.loads(lines[0])[0]
    dropped_list = []
    alpha = 0.8
    for t, frame_tensor in enumerate(frame_list):
        dropped_tensor = frame_tensor.clone()
        if str(t//2) in dp:
            dp_t = dp[str(t//2)]
            for i, j in dp_t:
                h_start, h_end = i * 28, (i + 1) * 28
                w_start, w_end = j * 28, (j + 1) * 28
                region = dropped_tensor[:, h_start:h_end, w_start:w_end]
                dropped_tensor[:, h_start:h_end, w_start:w_end] = region * (1-alpha) + 255 * alpha
        dropped_list.append(dropped_tensor)
    
    frame_list = frame_list[::2]
    dropped_list = dropped_list[::2]
    
    output_dir = "/home/gaohuan03/yaolinli/code/tokendrop_eval/case/sample_151_dropped_frames"
    for t, frm in enumerate(dropped_list):
        frm_img = to_pil_image(frm)
        frm_img.save(os.path.join(output_dir, f"{t:03d}second.png"), dpi=(300, 300))
    # also save the raw frames
    for t, frm in enumerate(frame_list):
        frm_img = to_pil_image(frm)
        frm_img.save(os.path.join(output_dir, f"{t:03d}second_raw.png"), dpi=(300, 300))

if __name__ == '__main__':
    # get_visual_case(
    #     dp_jsonl="/home/gaohuan03/liyicheng/code/Qwen2.5-VL/case/dp-case/px0d1_752.jsonl",
    #     video_path="/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding/sample_151/tmp_60/video_0_120.mp4",
    # )
    # get_visual_case(
    #     dp_jsonl="/home/gaohuan03/liyicheng/code/Qwen2.5-VL/case/dp-case/ft0d4_752.jsonl",
    #     video_path="/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding/sample_151/tmp_60/video_0_120.mp4",
    # )
    # get_visual_case(
    #     dp_jsonl="/home/gaohuan03/liyicheng/code/Qwen2.5-VL/case/dp-case/px0d1_671.jsonl",
    #     video_path="/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding/sample_135/tmp_60/video_0_187.mp4",
    # )
    # get_visual_case(
    #     dp_jsonl="/home/gaohuan03/liyicheng/code/Qwen2.5-VL/case/dp-case/ft0d4_671.jsonl",
    #     video_path="/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding/sample_135/tmp_60/video_0_187.mp4",
    # )
    
    # get_visual_case(
    #     dp_jsonl="/home/gaohuan03/liyicheng/code/Qwen2.5-VL/case/dp-case/42_px0d05.jsonl",
    #     video_path="/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding/sample_9/tmp_60/video_0_54.mp4"
    # )
    
    get_visual_case(
        dp_jsonl="/home/gaohuan03/liyicheng/code/Qwen2.5-VL/case/dp-case/sample_151_ft0d4.jsonl",
        video_path="/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding/sample_151/video.mp4"
    )