export CUDA_VISIBLE_DEVICES=0

RUN_NAME=feature_0d5
DROP_METHOD=feature     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.5

CKPT_PATH="wyccccc/TimeChatOnline-7B"
TASK_CSV="/home/gaohuan03/yaolinli/datasets/StreamingBench/annos/Real_Time_Visual_Understanding.csv"
VIDEO_DIR="/home/gaohuan03/liyicheng/Datasets/StreamingBench/Real-Time Visual Understanding"
RESULT_DIR="eval/result_streamingbench"

python eval/streamingbench.py \
    --run_name $RUN_NAME \
    --drop_method $DROP_METHOD \
    --drop_threshold $DROP_THRESHOLD \
    --ckpt_path "$CKPT_PATH" \
    --task_csv "$TASK_CSV" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"
