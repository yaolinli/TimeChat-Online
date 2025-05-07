export CUDA_VISIBLE_DEVICES=0

TASK_PARQUET="/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="/pfs/Datasets/Video-MME/origin_data/videos/data/"

RUN_NAME=feature_0d5
CKPT_PATH="wyccccc/TimeChatOnline-7B"
RESULT_DIR="eval/result_videomme"

# DTD arguments
DROP_METHOD=feature     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.5

python eval/scripts/videomme.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"
