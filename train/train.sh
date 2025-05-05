export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_PIXELS=200704
export VIDEO_MAX_PIXELS=200704


swift sft \
    --model_type qwen2_5_vl\
    --save_only_model \
    --save_strategy epoch \
    --model Qwen2.5-VL-7B-Instruct \
    --dataset datasets/timeit-v2/anno_data/qa_pairs_50k_segs_swift_sft.jsonl \
    datasets/tarsier2-recap-585k/tarsier2_recap_filtered_130k.jsonl \
    dataset/llava-video-178k/train_100k_64_frames_vaild.jsonl \
    datasets/timeit-v2/anno_data/qa_pairs_v2_24k_segs_swift_sft.jsonl \
    dataset/ours/v3_25k_options.jsonl dataset/ours/v3_20k_positive_options.jsonl \
    dataset/ours/v3_20k_negative_options.jsonl dataset/video-chat-flash/htstep_eventunderstanding-longvideo_annos-htstep_eventunderstanding_1k_1k.jsonl \
    dataset/video-chat-flash/ego4dhcap_eventunderstanding-longvideo_annos-ego4dhcap_eventunderstanding_2k_2k.jsonl \
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
