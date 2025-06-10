import os
import subprocess
import gradio as gr
import numpy as np
import time
import math
import torch
import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils.src.qwen_vl_utils import process_vision_info
from gradio_client import utils as client_utils
from argparse import ArgumentParser
from transformers import AutoProcessor
import tempfile

# Constants
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MIN_FRAMES = 4
MAX_FRAMES = 1016
MINUTE_FRAMES = 60
FPS = 1
FRAME_TOKEN = 143
DROP_METHOD = "feature"
DROP_THRESHOLD = 0.25
DROP_ABSOLUTE = True
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv')
PREFIX_TOKEN_LENGTH = 14
VIDEO_ADD_TOKEN_LENGTH = 1
SUFFIX_TOKEN_LENGTH = 5
FRAME_EACH_PROCESS = 2

def convert_webm_to_mp4(webm_path):
    mp4_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    mp4_path = mp4_file.name
    mp4_file.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", webm_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        mp4_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode())
        raise
    return mp4_path

def get_video_duration(video_path: str) -> float:
    if not video_path:
        return 0.0
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(result.stdout)
    except Exception:
        return 0.0
    
def concat_two_mp4s(mp4_a, mp4_b):
    list_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
    list_file.write(f"file '{mp4_a}'\nfile '{mp4_b}'\n")
    list_file.flush()
    list_file.close()

    out_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file.name,
        "-c", "copy",
        out_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    os.remove(list_file.name)
    os.remove(mp4_b)
    os.remove(mp4_a)
    return out_mp4

def extract_clip(video_path: str, begin: float, end: float) -> str:
    duration = end - begin
    base, ext = os.path.splitext(video_path)
    clip_path = f"{base}_clip_{int(begin*1000)}-{int(end*1000)}ms{ext}"
    cmd = [
        'ffmpeg','-y',
        '-ss', str(begin),
        '-t', str(duration),
        '-i', video_path,
        '-c', 'copy',
        '-avoid_negative_ts', '1',
        clip_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return clip_path

# Model & Processor loader
def _load_model_processor(args):
    device_map = 'cpu' if args.cpu_only else 'auto'
    if args.flash_attn2:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path,
            torch_dtype='auto',
            attn_implementation='flash_attention_2',
            device_map=device_map
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path,
            torch_dtype='auto',
            device_map=device_map
        )
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    return model, processor

# Main demo launcher
def _launch_demo(args, model, processor):
    default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team...'
    language = args.ui_language

    def get_text(text: str, cn_text: str):
        return text if language == 'en' else cn_text if language == 'zh' else text

    # Format history, embedding actual bytes for video and image
    def format_history(history: list):
        messages = []
        for item in history:
            if item['role'] == 'user':
                content_list = []
                if isinstance(item['content'], str):
                    content_list.append({'type':'text','text':item['content']})
                else:
                    for content in item['content']:
                        if isinstance(content, str) and content.split('.')[-1] in ['mp4','avi','mov']:
                            content_list.append({
                                'type':'video','video':content,
                                'min_pixels':MIN_PIXELS,'max_pixels':MAX_PIXELS,
                                'min_frames':MIN_FRAMES,'max_frames':MAX_FRAMES,
                                'fps':FPS
                            })
                        else:
                            content_list.append({'type':'image','image':content})
                messages.append({'role':'user','content':content_list})
        return messages

    def predict(inputs, visual_embedded, visual_embedded_type, keep_mask, begin_token_idx=None):
        out_ids = model.generate(**inputs, 
                                 visual_embedded=visual_embedded, 
                                 visual_embedded_type=visual_embedded_type, 
                                 max_new_tokens=128, 
                                 keep_mask=keep_mask,
                                 drop_method=DROP_METHOD,
                                 drop_threshold=DROP_THRESHOLD,
                                 drop_absolute=DROP_ABSOLUTE,
                                 begin_token_idx=begin_token_idx)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        resp = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        yield {'type':'text','data':resp}

    def process_drop_ratios(drop_ratio_list):
        n = len(drop_ratio_list)
        if n == 0:
            return []
        k = math.ceil(n * 0.1)
        
        indexed_elements = list(enumerate(drop_ratio_list))
        sorted_elements = sorted(indexed_elements, key=lambda x: x[1])
        
        selected_indices = [idx for idx, _ in sorted_elements[:k]]
        result = [(idx + 1) * FRAME_EACH_PROCESS for idx in selected_indices]
        result_sorted = sorted(result)
        return result_sorted

    def visual_embedding(video):
        inputs = []
        duration = 0.0
        if video :
            inputs.append({'role':'user','content':(video,)})
            duration = get_video_duration(video)
        formatted = format_history(inputs)
        image_inputs, video_inputs = process_vision_info(formatted)
        text = processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors='pt'
        ).to(model.device)
        visual_embedded, visual_embedded_type = model.embedding_visual_info(
            **inputs
        )
        keep_mask, drop_ratio_list = model.pre_token_drop(
                **inputs,
                max_new_tokens=128,
                visual_embedded=visual_embedded,
                visual_embedded_type=visual_embedded_type,
                drop_method=DROP_METHOD,
                drop_threshold=DROP_THRESHOLD,
                drop_absolute=DROP_ABSOLUTE
            )
        trigger_time = process_drop_ratios(drop_ratio_list)
        return visual_embedded, visual_embedded_type, keep_mask, trigger_time, duration

    def visual_embedding_stream(webm_path, pre_visual_embedded, pre_visual_embedded_type, pre_keep_mask, pre_mp4_path_state):
        if isinstance(webm_path, str) and webm_path.endswith(".webm"):
            mp4_path = convert_webm_to_mp4(webm_path)
        else:
            mp4_path = webm_path

        inputs = []
        if mp4_path:
            inputs.append({'role':'user','content':(mp4_path,)})
            formatted = format_history(inputs)
            image_inputs, video_inputs = process_vision_info(formatted)
            text = processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors='pt'
            ).to(model.device)
            visual_embedded, visual_embedded_type = model.embedding_visual_info(
                **inputs
            )
            keep_mask, _ = model.pre_token_drop(
                    **inputs,
                    max_new_tokens=32,
                    visual_embedded=visual_embedded,
                    visual_embedded_type=visual_embedded_type,
                    drop_method=DROP_METHOD,
                    drop_threshold=DROP_THRESHOLD,
                    drop_absolute=DROP_ABSOLUTE
                )
            if pre_visual_embedded is not None and pre_keep_mask is not None:
                visual_embedded = torch.concat([pre_visual_embedded, visual_embedded],dim=0)
                keep_mask = torch.concat([pre_keep_mask[:-SUFFIX_TOKEN_LENGTH+VIDEO_ADD_TOKEN_LENGTH], keep_mask[PREFIX_TOKEN_LENGTH+VIDEO_ADD_TOKEN_LENGTH:]],dim=0)
            if pre_mp4_path_state is not None:
                mp4_path_state = concat_two_mp4s(pre_mp4_path_state, mp4_path)
            else:
                mp4_path_state = mp4_path
        else:
            visual_embedded, visual_embedded_type, keep_mask, mp4_path_state = pre_visual_embedded, pre_visual_embedded_type, pre_keep_mask, pre_mp4_path_state

        return visual_embedded, visual_embedded_type, keep_mask, mp4_path_state

    def chat_predict(text, video, visual_embedded, visual_embedded_type, keep_mask, timestamp, history):
        inputs_his = []
        if video:
            clip = extract_clip(video, 0.0, round(float(timestamp), 1))
            inputs_his.append({'role':'user','content': (clip,)})
        if text:
            inputs_his.append({'role':'user','content': text})
            history.append({'role':'user','content': text})
            #"[Time: {timestamp}s]: "
        yield history
        formatted = format_history(inputs_his)
        text_input = processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(formatted)
        #local_video_tensor = [video_tensor[0][:int(timestamp)]]
        inputs = processor(
            text=[text_input], images=None, videos=video_inputs,
            padding=True, return_tensors='pt'
        ).to(model.device)

        #local_token_num = int(timestamp) * FPS * FRAME_TOKEN
        #visual_embedded = visual_embedded[:local_token_num]
        start_time = time.perf_counter()
        for chunk in predict(inputs, visual_embedded, visual_embedded_type, keep_mask):
            if chunk['type'] == 'text':
                elapsed = time.perf_counter() - start_time
                cur_ts = float(timestamp) + elapsed
                annotated = f"[Time: {cur_ts:.1f}s]: {chunk['data']}"
                history.append({'role':'assistant','content': annotated})
                yield history
        yield history

    def chat_predict_stream(text, mp4_path, visual_embedded, visual_embedded_type, keep_mask, history):
        inputs_his = []
        if mp4_path:
            inputs_his.append({'role':'user','content': (mp4_path,)})
        if text:
            inputs_his.append({'role':'user','content': text})
            history.append({'role':'user','content': text})
        yield history

        formatted = format_history(inputs_his)
        text_input = processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(formatted)
        #local_video_tensor = [video_tensor[0][:int(timestamp)]]
        inputs = processor(
            text=[text_input], images=None, videos=video_inputs,
            padding=True, return_tensors='pt'
        ).to(model.device)

        #local_token_num = int(timestamp) * FPS * FRAME_TOKEN
        #visual_embedded = visual_embedded[:local_token_num]
        for chunk in predict(inputs, visual_embedded, visual_embedded_type, keep_mask):
            if chunk['type'] == 'text':
                annotated = chunk['data']
                history.append({'role':'assistant','content':annotated})
                yield history
        yield history

    with gr.Blocks() as demo, ms.Application(), antd.ConfigProvider():
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label='System Prompt', value=default_system_prompt)
        with gr.Tabs():
            with gr.Tab('Single-Round Response'):
                with gr.Row():
                    video_input = gr.Video(sources=['upload'], label='Upload Video', elem_id="my_video_1")
                    chatbot = gr.Chatbot(type='messages')
                with gr.Row():
                    time_slider = gr.Slider(
                        label='Current Video Time (s)',
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0,
                        interactive=False,
                        elem_id="my_slider_1"
                    )

                visual_embedded = gr.State()
                visual_embedded_type = gr.State()
                keep_mask = gr.State()
                trigger_time = gr.State()
                video_duration = gr.State(0.0)

                video_input.upload(
                    fn=visual_embedding,
                    inputs=[video_input],
                    outputs=[visual_embedded, visual_embedded_type, keep_mask, trigger_time, video_duration]
                )

                video_input.change(
                    fn=lambda v: gr.update(maximum=get_video_duration(v), value=0),
                    inputs=[video_input],
                    outputs=[time_slider],
                    js="""
                    (v) => {
                        setTimeout(() => {
                            const videoElement = document.querySelector('#my_video_1 video');
                            const sliderInput = document.querySelector('#my_slider_1 input[type="range"]');
                            if (videoElement && sliderInput) {
                                videoElement.removeEventListener('timeupdate', window.videoTimeUpdateHandler_1);
                                window.videoTimeUpdateHandler_1 = function() {
                                    sliderInput.value = videoElement.currentTime;
                                    sliderInput.dispatchEvent(new Event('input', { bubbles: true }));
                                };
                                videoElement.addEventListener('timeupdate', window.videoTimeUpdateHandler_1);
                            }
                        }, 100);
                        return v;
                    }
                    """
                )

                text_input = gr.Textbox(show_label=False, placeholder='Enter question...')
                with gr.Row():
                    submit_btn = gr.Button('Submit Question', variant='primary', size='lg')
                    clear_btn = gr.Button('Clear History', size='lg')
                def clear_chat():
                    return [], gr.update(value=None)

                submit_btn.click(
                    fn=chat_predict,
                    inputs=[text_input, video_input, visual_embedded, visual_embedded_type, keep_mask, time_slider, chatbot],
                    outputs=[chatbot]
                )
                clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, text_input])

            with gr.Tab('Live Camera'):
                with gr.Row():
                    with gr.Column():
                        camera_video = gr.Video(
                            sources=['webcam'],
                            streaming=False,
                            label="Live Camera Video",
                            elem_id="my_cam_video"
                        )
                        stream_status = gr.Textbox(visible=False, show_label=False)
                        video_status = gr.Textbox(visible=False, show_label=False, lines=2)
                    chatbot = gr.Chatbot(type='messages')

                visual_embedded = gr.State()
                visual_embedded_type = gr.State()
                keep_mask = gr.State()
                mp4_path_state = gr.State(None)

                text_input = gr.Textbox(show_label=False, placeholder='Enter question...')
                with gr.Row():
                    submit_video_btn = gr.Button("Submit Video", variant='primary', size='lg')
                    submit_btn = gr.Button('Submit Question', variant='primary', size='lg')
                    clear_btn = gr.Button('Clear History', size='lg')
                
                def middle_status():
                    return gr.update(value="The video is streaming and will take some time. Please be patient.", visible=True)
                def final_status():
                    return gr.update(value=None, visible=False)
                def handle_video_submit(video, ve, ve_type, mask, path):
                    yield None, None, None, None, gr.update(value="Processing video...", visible=True)
                    ve, ve_type, mask, path = visual_embedding_stream(video, ve, ve_type, mask, path)
                    yield ve, ve_type, mask, path, gr.update(value="✅ The video has been processed. If you want to continue uploading videos, click \n“Submit Video” again after uploading new video.", visible=True)
                
                submit_video_btn.click(
                    fn=middle_status,
                    inputs=[],
                    outputs=[stream_status]
                ).then(
                    fn=handle_video_submit,
                    inputs=[camera_video, visual_embedded, visual_embedded_type, keep_mask, mp4_path_state],
                    outputs=[visual_embedded, visual_embedded_type, keep_mask, mp4_path_state, video_status]
                ).then(
                    fn=final_status,
                    inputs=[],
                    outputs=[stream_status]
                )
                
                def clear_chat():
                    return [], gr.update(value=None)
                submit_btn.click(
                    fn=chat_predict_stream,
                    inputs=[text_input, mp4_path_state, visual_embedded, visual_embedded_type, keep_mask, chatbot],
                    outputs=[chatbot]
                )
                clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, text_input])
        
    demo.launch(share=args.share, inbrowser=args.inbrowser, server_port=args.server_port, server_name=args.server_name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', default='Qwen2.5-VL-7B-Instruct', type=str)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--flash-attn2', default=True, action='store_true')
    parser.add_argument('--share', default=False, action='store_true')
    parser.add_argument('--inbrowser', action='store_true')
    parser.add_argument('--server-port', type=int, default=7890)
    parser.add_argument('--server-name', type=str, default='127.0.0.1')
    parser.add_argument('--ui-language', choices=['en','zh'], default='zh')
    args = parser.parse_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)
