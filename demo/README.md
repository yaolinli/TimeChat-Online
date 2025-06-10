# Video-QA Web Demo

## Functions
- **Single-Round Video Interaction**: Upload a video, seek to any timestamp, and ask questions about the current frame.

- **Live Camera Stream**: Use your local webcam for real-time video Q&A.

- **Frame Sampling & Token Drop**: Built-in drop ratios optimize inference speed by discarding less relevant frames.

- **Multi-Format Support**: Compatible with .mp4, .mov, .avi, .mkv and mixed video inputs.

## Requirements
Make sure you have **Python 3.8+** installed for compatibility with the following dependencies.
```bash
pip install -r requirements_demo.txt
```
If your system does not have **ffmpeg**, you may need to install it manually:
```bash
sudo apt install ffmpeg
```
We recommend that you also install **flash_attn_2**.
```bash
pip install flash-attn --no-build-isolation
```

## Usage

Run the web demo script with optional parameters:
```bash
python web_demo.py [options]
```

### Options
| Option                   | Description                       | Default                   |
|--------------------------|-----------------------------------|---------------------------|
| `-c`, `--checkpoint-path`| Path to the model checkpoint      | `Qwen2.5-VL-7B-Instruct`*  |
| `--cpu-only`             | Run on CPU only                   | `False`                   |
| `--flash-attn2`          | Enable FlashAttention v2          | `True`                    |
| `--share`                | Enable sharing via public link    | `False`                   |
| `--inbrowser`            | Open UI in default web browser    | `False`                   |
| `--server-port`          | Port number for the server        | `7890`                    |
| `--server-name`          | Hostname/IP for the server        | `127.0.0.1`               |
| `--ui-language`          | UI language (`en` or `zh`)        | `zh`                      |

*For a better experience, we recommend that you change the default model path to our released checkpoint [`wyccccc/TimeChatOnline-7B`](https://huggingface.co/wyccccc/TimeChatOnline-7B).

## Demo workflow
![demo](assets/demo.png)
Navigate to http://\<server-name\>:\<server-port\>.
- **Single-Round Response**
Switch to  Single-Round Response tab.
Upload and play videos → Drag the time slider(or wait for automatic updates) → Type a question → Click Submit Question.

- **Live Camera**
Switch to Live Camera tab.
Click Submit Video to capture webcam frames → Type a question → Click Submit Question.