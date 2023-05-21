from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
from datasets import load_dataset
import os


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)


@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    train_epochs: int = field(default=4)
    train_file: str = field(default="merge_data/only_mental_0426.json")

@dataclass
class DataArguments:
    max_source_length: int = field(default=1024)
    max_target_length: int = field(default=256)
    max_train_samples: int = field(default=None)
    ignore_pad_token_for_loss: bool = field(default=True)
    overwrite_cache: bool = field(default=False)
    prompt_column: str = field(default='input')
    response_column: str = field(default='output')
    history_column: str = field(default=None)
    preprocessing_num_workers: int = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_prefix: str = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

writer = SummaryWriter()
finetune_args, training_args, data_args = HfArgumentParser(
    (FinetuneArguments, TrainingArguments, DataArguments)
).parse_args_into_dataclasses()
training_args.num_train_epochs = finetune_args.train_epochs
training_args.save_strategy = 'epoch'
training_args.do_train=True
training_args.gradient_accumulation_steps = 8
training_args.max_steps = 0
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = len(ids)
        # seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[data_args.prompt_column])):
            if examples[data_args.prompt_column][i] and examples[data_args.response_column][i]:
                query, answer = examples[data_args.prompt_column][i], examples[data_args.response_column][i]

                if data_args.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[data_args.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = data_args.source_prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]

                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position+1:]
                
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

def main():
    

    # init model
    model = AutoModel.from_pretrained(
        "THUDM/chatglm-6b", load_in_8bit=False, trust_remote_code=True, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    # Load dataset
    
    # load dataset
    dataset = load_dataset(
        'json',
        data_files=finetune_args.train_file
    )
    if data_args.max_train_samples is not None:
            max_train_samples = min(len(dataset), data_args.max_train_samples)
            dataset = dataset.select(range(max_train_samples))
    dataset = dataset['train']
    column_names = dataset.column_names
    # print(column_names,dataset.keys())
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    # print_dataset_example(train_dataset[0])

    # dataset = datasets.load_from_disk(finetune_args.dataset_path)
    print(f"\n{len(dataset)=}\n")
    print(training_args)
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
