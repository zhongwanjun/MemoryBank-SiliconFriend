from transformers.generation.logits_process import LogitsProcessor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    set_seed,
)
from peft import PeftModel
import torch, os

max_chunk_overlap = 20
pre_seq_len = 128
prefix_projection = False

def load_chatglm_tokenizer_and_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model = model.eval()
    return tokenizer, model

def load_lora_chatglm_tokenizer_and_model(model_path,adapter_model):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_model)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = model.half().cuda().eval()
    return tokenizer, model

def load_prefix_chatglm_tokenizer_and_model(model_path,ptuning_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.pre_seq_len = pre_seq_len
    config.prefix_projection = prefix_projection
    model = AutoModel.from_pretrained(model_path, config=config,trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.cuda().eval()

    if pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.float()
    return tokenizer, model

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores