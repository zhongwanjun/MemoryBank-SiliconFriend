export OPENAI_API_KEY=API_KEY
base_model=THUDM/chatglm-6b
adapter_model=YOUR_LORA_ADAPTER_PATH
python app_demo.py \
    --base_model $base_model \
    --adapter_model $adapter_model \
    --language cn

