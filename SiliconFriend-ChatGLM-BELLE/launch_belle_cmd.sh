export OPENAI_API_KEY=OPENAI_API_KEY
base_model=PATH_TO_BELLE_MODEL(https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)
adapter_model=PATH_TO_BELLE_ADAPTER_MODEL
python cli_ptuning_memory_search_langchain.py \
    --model_type belle \
    --base_model $base_model \
    --adapter_model $adapter_model  \
    --language cn \
    --enable_forget_mechanism False \
    --memory_basic_dir ../../memories \
    --memory_file update_memory.json \