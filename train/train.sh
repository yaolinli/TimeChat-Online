export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_PIXELS=200704
export VIDEO_MAX_PIXELS=200704


swift sft \
    --model_type qwen2_5_vl\
    --save_only_model \
    --save_strategy epoch \
    --model Qwen2.5-VL-7B-Instruct \
    --dataset llava_video_100k.jsonl tarsier2_129k.jsonl videochat_flash_3k.jsonl time_chat_online_139k_train.jsonl \
    --enable_cache true \
    --freeze_vit true \
    --freeze_aligner false \
    --logging_steps 20 \
    --learning_rate 1e-5 \
    --output_dir  ckpt/qwen2_5/fourdataset_v2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_train_epochs 1 \
    --attn_impl flash_attn \
    --train_type full \
    --eval_steps 200 \
    --save_steps 1000 \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --max_length 11264 \
    --warmup_ratio 0.05 \
    --truncation_strategy delete
