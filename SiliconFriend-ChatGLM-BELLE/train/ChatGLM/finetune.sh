python finetune.py \
    --train_file only_mental_0426.json \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --fp16 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output