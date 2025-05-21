export CUDA_VISIBLE_DEVICES=0

TASK_JSON="/home/gaohuan03/liyicheng/code/OVO-Bench/data/ovo_bench_new.json"
VIDEO_DIR="/home/gaohuan03/liyicheng/code/OVO-Bench/data"

RUN_NAME=feature_0d5
CKPT_PATH="wyccccc/TimeChatOnline-7B"
RESULT_DIR="eval/result_ovobench"

# DTD arguments
DROP_METHOD=feature     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.5

python eval/scripts/ovobench.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"
