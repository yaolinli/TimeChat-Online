export CUDA_VISIBLE_DEVICES=0

RUN_NAME=feature_0d5
DROP_METHOD=feature     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.5

CKPT_PATH="/pfs/Models/Qwen2.5-VL-7B-Instruct"
TASK_PARQUET="/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="/pfs/Datasets/Video-MME/origin_data/videos/data/"
RESULT_DIR="eval/result_videomme"

python eval/videomme.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"
