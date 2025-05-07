export CUDA_VISIBLE_DEVICES=0,1

TASKS=egoschema
NUM_PROCESSES=2
MAIN_PROCESS_PORT=29501

RUN_NAME=feature_0d5
CKPT_PATH=wyccccc/TimeChatOnline-7B
RESULT_DIR=eval/result_egoschema

# DTD arguments
DROP_METHOD=feature     # "feature" or "pixel" or "none"
DROP_TRESHOLD=0.5

python -m accelerate.commands.launch \
    --num_processes $NUM_PROCESSES \
    --main_process_port $MAIN_PROCESS_PORT \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=${CKPT_PATH},use_flash_attention_2=True,min_pixels=448*448,max_pixels=448*448,max_num_frames=1016,fps=1,drop_method=${DROP_METHOD},drop_threshold=${DROP_TRESHOLD},dr_save_path=${RESULT_DIR}/drop/${RUN_NAME}.jsonl \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path ${RESULT_DIR}/log
