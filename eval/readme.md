# Evaluation

## Requirements

```bash
pip install transformers==4.50.0 accelerate==1.5.2 qwen-vl-utils[decord]==0.0.10
```

For StreamingBench, `ffmpeg` is needed:

```bash
pip install ffmpeg-python==0.2.0 
```

For OVO-Bench, `moviepy` is needed:

```bash
pip install moviepy==1.0.3
```

For EgoSchema, LongVideoBench and MLVU, a modified version of `lmms-eval` should be installed here:

```bash
pip install -e ./eval
```

## Benchmarks

### StreamingBench

Download the dataset from [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench). Only `StreamingBench/Real_Time_Visual_Understanding.csv` and `Real-Time Visual Understanding_*.zip` is needed.  

In [`eval_streamingbench.sh`](./scripts/eval_streamingbench.sh), set `TASK_CSV` to the path of `StreamingBench/Real_Time_Visual_Understanding.csv`, and `VIDEO_DIR` to the directory of videos unzipped from `Real-Time Visual Understanding_*.zip`. Then run:

```bash
bash eval/scripts/eval_streamingbench.sh
```

### Video-MME

Download the dataset from [lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME).  

In [`eval_videomme.sh`](./scripts/eval_videomme.sh), set `TASK_PARQUET` to the path of `videomme/test-00000-of-00001.parquet`, and `VIDEO_DIR` to the directory of videos unzipped from `videos_chunked_*.zip`. Then run:

```bash
bash eval/scripts/eval_videomme.sh
```

### OVO-Bench

Download videos from [JoeLeelyf/OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) on HuggingFace. Only `src_videos.tar.parta[a-e]` is needed, since our code includes chunking videos.  

Download [`ovo_bench_new.json`](https://github.com/JoeLeelyf/OVO-Bench/blob/main/data/ovo_bench_new.json) from [JoeLeelyf/OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench) on Github.  

In [`eval_ovobench.sh`](./scripts/eval_ovobench.sh), set `TASK_JSON` to the path of `ovo_bench_new.json`, and `VIDEO_DIR` to the directory of videos untarred from `src_videos.tar.parta[a-e]`. Then run:

```bash
bash eval/scripts/eval_ovobench.sh
```
### EgoSchema / LongVideoBench / MLVU

```bash
bash eval/scripts/eval_egoschema.sh
bash eval/scripts/eval_longvideobench.sh
bash eval/scripts/eval_mlvu.sh
```

## Arguments

In each `eval_*.sh`, there are several alterable arguments:

- `RUN_NAME`: The name of this run, used for logging.  

- `CKPT_PATH`: The path to the checkpoint for evaluation. Default to our released checkpoint [`wyccccc/TimeChatOnline-7B`](https://huggingface.co/wyccccc/TimeChatOnline-7B).  

- `RESULT_DIR`: The directory used to store logs, outputs and other infos. It will be auto-created if not exists.  

- `DROP_METHOD`: The DTD method used. It can be set to `feature` for feature-level drop, `pixel` for pixel-level drop, and `none` to avoid drop.  

- `DROP_THRESHOLD`: The threshold used for drop.  
    - For feature-level drop, it ranges in [-1, 1]. The drop ratio grows as the threshold decreases.  
    - For pixel-level drop, it ranges in [0, 1]. The drop ratio grows as the threshold increases.  

For details of the drop process, you can refer to the code [here](./qwen2_5_vl/modeling_qwen2_5_vl_DTD.py#L1137-L1400).  
