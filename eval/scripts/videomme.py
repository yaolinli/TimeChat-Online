from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Parameters
RUN_NAME = "feature_0d5"
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "wyccccc/TimeChatOnline-7B"

TASK_PARQUET = "/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "/pfs/Datasets/Video-MME/origin_data/videos/data/"
RESULT_DIR = "eval/result_videomme"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MAX_FRAMES = 900
MIN_FRAMES = 4
FPS = 1

# Prompt template
prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. 
{}
Options: {}
The best answer is:"""

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

## helper functions
def time_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]

### Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true") # Default is absolute
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    
    # Update global variables
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_PARQUET = args.task_parquet
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    LOG_PATH = LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time)
    OUTPUT_JSONL = OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time)
    DR_SAVE_PATH = DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time)
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MAX_FRAMES = args.max_frames
    MIN_FRAMES = args.min_frames
    FPS = args.fps
    
    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)

    # Add file handler
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Print run info
    logger.info(f"Running {RUN_NAME} on Video-MME")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task parquet: {TASK_PARQUET}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"FPS: {FPS}")
    
    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    ## Use Qwen2.5-VL
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model and processor from {CKPT_PATH}")

    # Load task info
    task_df = pd.read_parquet(TASK_PARQUET)

    # Inference
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            video_id, duration, domain, sub_category, url, videoID, question_id, task_type, question, options, answer = \
                row.video_id, row.duration, row.domain, row.sub_category, row.url, row.videoID, row.question_id, row.task_type, row.question, row.options, row.answer
            video_path = osp.join(VIDEO_DIR, videoID + '.mp4')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "min_pixels": MIN_PIXELS,
                            "max_pixels": MAX_PIXELS,
                            "max_frames": MAX_FRAMES,
                            "min_frames": MIN_FRAMES,
                            "fps": FPS
                        },
                        {
                            "type": "text", 
                            "text": prompt.format(question, '\n'.join(options.tolist()))
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(torch.device('cuda'))
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                drop_method=DROP_METHOD,
                drop_threshold=DROP_THRESHOLD,
                drop_absolute=DROP_ABSOLUTE,
                dr_save_path=DR_SAVE_PATH,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
            cnt_total['overall'] += 1
            cnt_total[duration] += 1
            if extract_characters_regex(response) == answer:
                cnt_correct['overall'] += 1
                cnt_correct[duration] += 1

            output_dict = {
                'video_id': video_id,
                'duration': duration,
                'domain': domain,
                'sub_category': sub_category,
                'url': url,
                'videoID': videoID,
                'question_id': question_id,
                'task_type': task_type,
                'question': question,
                'options': options.tolist(), # convert numpy array to list for json serialization
                'answer': answer,
                'response': response
            }
            with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
        except Exception as e:
            logger.error(f"Error in processing {row}: {e}")
        # break
    end_time = time.time()
    cost_time = int(end_time - start_time)

    # Print results
    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100*cnt_correct['overall']/cnt_total['overall']:.1f}%")
    for duration in task_df.duration.unique():
        if cnt_total[duration] == 0:
            continue
        logger.info(f"Duration {duration}: {cnt_correct[duration]}/{cnt_total[duration]} = {100*cnt_correct[duration]/cnt_total[duration]:.1f}%")

    # Collect drop ratio info
    if DROP_METHOD is not None and DR_SAVE_PATH is not None:
        drop_list, total_list, ratio_list = [], [], []
        with open(DR_SAVE_PATH, 'r') as f:
            lines = f.readlines()
        for line in lines:
            drop_ratio_info = json.loads(line)
            drop_list.append(drop_ratio_info['drop'])
            total_list.append(drop_ratio_info['total'])
            ratio_list.append(drop_ratio_info['ratio'])
        total_dr = sum(drop_list) / sum(total_list)
        avg_dr = sum(ratio_list) / len(ratio_list)
        logger.info(f"Total drop ratio (weighted drop ratio): {100*total_dr:.1f}%")
        logger.info(f"Average drop ratio (unweighted drop ratio): {100*avg_dr:.1f}%")

    # Print time
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
