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
from moviepy.editor import VideoFileClip
import math
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Parameters
RUN_NAME = "feature_0d5"
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "wyccccc/TimeChatOnline-7B"

TASK_JSON = "/home/gaohuan03/liyicheng/code/OVO-Bench/data/ovo_bench_new.json"
VIDEO_DIR = "/home/gaohuan03/liyicheng/code/OVO-Bench/data"
RESULT_DIR = "eval/result_ovobench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MIN_FRAMES = 4
MAX_FRAMES = 720
FPS = 1
# NFRAMES = 64

backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

# helper functions
def build_prompt(task, question, options, _anno_, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
        prompt = f"""
            Question: {question}
            Options:
            {formatted_options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
            Do not include any additional text or explanation in your response.
        """
    elif task == "REC":
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt = f""" 
            You're watching a video in which people may perform a certain type of action repetively. 
            The person performing this kind of action are referred to as 'they' in the following statement.
            You're task is to count how many times have different people in the video perform this kind of action in total.
            One complete motion counts as one. 
            Now, answer the following question: {question}
            Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
            Do not include any additional text or explanation in your response.
        """
    elif task == "SSR":
        step = _anno_["test_info"][index]["step"]
        prompt = f"""
            You're watching a tutorial video which contain a sequential of steps. 
            The following is one step from the whole procedures: 
            {step}
            Your task is to determine if the man or woman in the video is currently performing this step.
            Answer only with “Yes” or “No”.
            Do not include any additional text or explanation in your response.
        """

    elif task == "CRR":
        question = _anno_["question"]
        answer = _anno_["answer"]
        prompt = f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with “Yes” or “No”.
            Do not include any additional text or explanation in your response.
        """
    return prompt

def chunk_video(video_path, end_time, start_time=0):
    end_time = math.ceil(end_time)
    # 检查是否已经存在 chunked video
    video_name = osp.splitext(osp.basename(video_path))[0]
    output_dir = osp.join(VIDEO_DIR, "chunked")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    output_file = osp.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if osp.exists(output_file):
        logger.debug(f"Chunked video {output_file} already exists")
        return output_file
    # 如果不存在，进行分割
    video = VideoFileClip(video_path)
    # 确保 end_time 不超过视频长度
    if end_time > video.duration:
        end_time = video.duration
    clip = video.subclip(start_time, end_time)
    temp_audiofile = osp.join("tmp", f"temp_audio_{video_name}_{start_time}_{end_time}.mp3")
    clip.write_videofile(output_file, logger=None, temp_audiofile=temp_audiofile)
    logger.debug(f"Chunked video {output_file} saved")
    return output_file

def get_response(prompt, video_path, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "min_frames": MIN_FRAMES,
                    "max_frames": MAX_FRAMES,
                    "fps": FPS,
                    # "nframes": NFRAMES,
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
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
    return response

def score(results):
    def calculate_score_backward_realtime(results):
        def get_score(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        # Calculate Score for Every Result
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        
        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(results):
        def get_score_REC(response, gt):
            if response == None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)
        
        def get_score_SSR_CRR(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        
        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            # Calculate score for REC
            if result["task"] == "REC":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    # scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
                    cnt_correct += get_score_REC(test_info_["response"], test_info_["count"])
                scores["REC"].append(cnt_correct / len(result["test_info"]))
            # Calculate score for SSR
            if result["task"] == "SSR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        # scores["SSR"].append(1)
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    # scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["SSR"].append(cnt_correct / len(result["test_info"]))
            # Calculate score for CRR
            if result["task"] == "CRR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        # scores["CRR"].append(1)
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    # scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["CRR"].append(cnt_correct / len(result["test_info"]))
        return results, scores
    
    backward_results = results["backward"]
    realtime_results = results["realtime"]
    forward_results = results["forward"]
    avg_scores = {
        "backward": [],
        "realtime": [],
        "forward": []
    }

    if len(backward_results) > 0:
        # print("Evaluate Backward Tracing...")
        backward_results, backward_scores = calculate_score_backward_realtime(backward_results)
        # correct_backward, total_backward = 0, 0
        for k, v in backward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            # correct_backward += sum(v)
            # total_backward += len(v)
            avg_scores["backward"].append(sum(v)/len(v))
        # print(f"Backward Avg.: {100 * correct_backward / total_backward:.2f}\n")
        logger.info(f"Backward Avg.: {100 * sum(avg_scores['backward'])/len(avg_scores['backward']):.2f}\n")
    else:
        # correct_backward = 0
        # total_backward = 0
        pass
        
    if len(realtime_results) > 0:
        # print("Evaluate Real-time Visual Perception...")
        realtime_results, realtime_scores = calculate_score_backward_realtime(realtime_results)
        # correct_realtime, total_realtime = 0, 0
        for k, v in realtime_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            # correct_realtime += sum(v)
            # total_realtime += len(v)
            avg_scores["realtime"].append(sum(v)/len(v))
        # print(f"Realtime Avg.: {100 * correct_realtime / total_realtime:.2f}\n")
        logger.info(f"Realtime Avg.: {100 * sum(avg_scores['realtime'])/len(avg_scores['realtime']):.2f}\n")
    else:
        # correct_realtime = 0
        # total_realtime = 0
        pass

    if len(forward_results) > 0:
        # print("Evaluate Forward Active Responding...")
        forward_results, forward_scores = calculate_score_forward(forward_results)
        # correct_forward, total_forward = 0, 0
        for k, v in forward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            # correct_forward += sum(v)
            # total_forward += len(v)
            avg_scores["forward"].append(sum(v)/len(v))
        # print(f"Forward Avg.: {100 * correct_forward / total_forward:.2f}\n")
        logger.info(f"Forward Avg.: {100 * sum(avg_scores['forward'])/len(avg_scores['forward']):.2f}\n")
    else:
        # correct_forward = 0
        # total_forward = 0
        pass

    # logger.info(f"Total Avg.: {100 * (sum(avg_scores['backward']) + sum(avg_scores['realtime']) + sum(avg_scores['forward'])) / (len(avg_scores['backward']) + len(avg_scores['realtime']) + len(avg_scores['forward'])):.2f}")
    logger.info(f"Total Avg.: {100 * (sum(avg_scores['backward'])/len(avg_scores['backward']) + sum(avg_scores['realtime'])/len(avg_scores['realtime']) + sum(avg_scores['forward'])/len(avg_scores['forward'])) / 3:.2f}")

### Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true") # Default is absolute
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    # parser.add_argument("--nframes", type=int, default=NFRAMES)
    args = parser.parse_args()
    
    # Update global variables
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    FPS = args.fps
    # NFRAMES = args.nframes
    
    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)
    
    # Add file handler
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # Print run info
    logger.info(f"Running {RUN_NAME} on StreamingBench")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task json: {TASK_JSON}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"FPS: {FPS}")
    # logger.info(f"Number of frames: {NFRAMES}")
    
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
    with open(TASK_JSON, 'r') as f:
        task_list = json.load(f)
    
    # Inference
    start_time = time.time()
    for item in tqdm(task_list):
        try:
            if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                id, video, task, question, options, realtime, gt = \
                    item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']
                prompt = build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                )
                video_path = osp.join(VIDEO_DIR, video)
                chunk_video_path = chunk_video(video_path=video_path, end_time=realtime)
                response = get_response(prompt=prompt, video_path=chunk_video_path, model=model, processor=processor)
                
                output_dict = {
                    'id': id,
                    'video': video,
                    'task': task,
                    'question': question,
                    'response': response,
                    'ground_truth': chr(65 + gt),
                }
            
            elif item['task'] in forward_tasks:
                id, video, task, test_info = \
                    item['id'], item['video'], item['task'], item['test_info']
                for i in range(len(test_info)):
                    prompt = build_prompt(
                        task=task,
                        question=None,
                        options=None,
                        _anno_=item,
                        index=i,
                    )
                    realtime = test_info[i]['realtime']
                    video_path = osp.join(VIDEO_DIR, video)
                    chunk_video_path = chunk_video(video_path=video_path, end_time=realtime)
                    response = get_response(prompt=prompt, video_path=chunk_video_path, model=model, processor=processor)
                    item['test_info'][i]['response'] = response
                    
                output_dict = item
            
            with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
        except Exception as e:
            logger.error(f"Error in processing {item}: {e}")
        # break
    end_time = time.time()
    cost_time = int(end_time - start_time)
    
    # Print results
    results = defaultdict(list)
    with open(OUTPUT_JSONL, 'r') as f:
        lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        if item['task'] in backward_tasks:
            results['backward'].append(item)
        elif item['task'] in realtime_tasks:
            results['realtime'].append(item)
        elif item['task'] in forward_tasks:
            results['forward'].append(item)
    score(results)
    
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