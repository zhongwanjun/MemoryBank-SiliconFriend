# -*- coding:utf-8 -*-
import os, shutil
import logging
import sys
import time
import gradio as gr
import nltk
import torch
prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(prompt_path)
sys.path.append(bank_path)
from utils.prompt_utils import *
from utils.memory_utils import enter_name, summarize_memory_event_personality, save_local_memory
from utils.model_utils import load_chatglm_tokenizer_and_model,load_belle_tokenizer_and_model,load_lora_chatglm_tokenizer_and_model,load_prefix_chatglm_tokenizer_and_model, InvalidScoreLogitsProcessor
from transformers.generation.utils import LogitsProcessorList
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path
from memory_retrieval.configs.model_config import *
from utils.app_modules.utils import *
from utils.app_modules.presets import *
from utils.app_modules.overwrites import *
from utils.sys_args import data_args,model_args

 
memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
if not os.path.exists(memory_dir):
    json.dump({},open(memory_dir,"w",encoding="utf-8"))

language = data_args.language
if data_args.enable_forget_mechanism:
    from memory_retrieval.forget_memory import LocalMemoryRetrieval
else:
    from memory_retrieval.local_doc_qa import LocalMemoryRetrieval

local_memory_qa = LocalMemoryRetrieval()
EMBEDDING_MODEL = EMBEDDING_MODEL_CN if language == 'cn' else EMBEDDING_MODEL_EN
local_memory_qa.init_cfg(
                        embedding_model=EMBEDDING_MODEL,
                        embedding_device=EMBEDDING_DEVICE,
                        top_k=data_args.memory_search_top_k,
                        language=language)

meta_prompt = generate_meta_prompt_dict_chatglm_app()[language]
new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatglm()[language]
user_keyword = generate_user_keyword()[language]
ai_keyword = generate_ai_keyword()[language]
boot_name = boot_name_dict[language]
boot_actual_name = boot_actual_name_dict[language]

global memory
memory = json.loads(open(memory_dir,"r",encoding="utf-8").read())
# tokenizer, model= load_prefix_chatglm_tokenizer_and_model(base_model,adapter_model)
if model_args.model_type=='chatglm':
    tokenizer, model= load_lora_chatglm_tokenizer_and_model(model_args.base_model,model_args.adapter_model)
elif model_args.model_type=='belle':
    tokenizer, model= load_belle_tokenizer_and_model(model_args.base_model,model_args.adapter_model)

# tokenizer, model= load_chatglm_tokenizer_and_model(base_model)
# tokenizer, model, device = load_tokenizer_full_model(base_model, load_8bit=load_8bit)
# )

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

load_8bit = (
    sys.argv[3].startswith("8")
    if len(sys.argv) > 3 else False
)



def chat(model, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
            do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, 
            user_memory=None,
            user_name=None,
            user_memory_index=None,
            local_memory_qa=None,
            **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = { "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor,**kwargs}
    if isinstance(local_memory_qa,gr.State):
        local_memory_qa = local_memory_qa.value
    prompt = build_prompt_with_search_memory_chatglm_app(history,query,user_memory,user_name,user_memory_index,local_memory_qa,meta_prompt,new_user_meta_prompt,user_keyword,ai_keyword,boot_actual_name,language)
    print(prompt)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    if model_args.model_type == 'chatglm':
        response = model.process_response(response)
    response = clean_result(response,prompt,stop_words=[user_keyword])
    return response

def clean_result(result,prompt,stop_words):
    result = result.replace(prompt,"").strip() 
    result = result.replace("&nbsp;","")
    # if is_stop_word_or_prefix(result, stop_words) is False:
    # print(result) 
    for stop in stop_words:
        if stop in result:
            result = result[:result.index(stop)].strip()
    result = result.replace(ai_keyword,"").strip()
    result = result.replace(":","").strip()
    result = result.replace("：","").strip()
    # print(result)
    return convert_to_markdown(result)

def predict_new(
    text,
    chatbot,
    history, 
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    user_name,
    user_memory,
    memory,
    user_memory_index,
    local_memory_qa
):
    if text == "":
        yield chatbot, history, "Empty context."
        return
   
    if len(history) > data_args.value.max_history:
        history = history[-data_args.value.max_history:]
    # print(history)
    response = chat(model,tokenizer,text,history=history,
                    num_beams=1, 
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=1,
                    max_length=max_context_length_tokens,
                    max_new_tokens=max_length_tokens,
                    user_memory=user_memory,
                    user_name=user_name,
                    user_memory_index=user_memory_index,
                    local_memory_qa=local_memory_qa)
    result = response
    print(print('\n~~~~~~~~~~~~\nquestion: ',text,'\nResponse:',result,'\n----------------------\n'))
 
    torch.cuda.empty_cache()
    
    a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
                    [text, convert_to_markdown(result)]], history + [[text, result]]
    if user_name:
        memory = save_local_memory(memory,b,user_name,data_args)
    yield a, b, memory, "Generating..."
    
    if shared_state.interrupted:
        shared_state.recover()
        try:
            yield a, b, memory, "Stop: Success" 
            return
        except:
            pass
    try:
        yield a, b, memory, "Generate: Success"
    except:
        pass


def retry(
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    user_name,
    user_memory,
    memory,
    user_memory_index,
    local_memory_qa
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict_new(
        text=inputs,
        chatbot=chatbot,
        history=history,
        top_p=top_p,
        temperature=temperature,
        max_length_tokens=max_length_tokens,
        max_context_length_tokens=max_context_length_tokens,
        user_name=user_name,
        user_memory=user_memory,
        memory=memory,
        user_memory_index=user_memory_index,
        local_memory_qa=local_memory_qa
    ):
        yield x

gr.Chatbot.postprocess = postprocess

with open(f"{path}/utils/assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()



with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    history = gr.State([])
    user_name = gr.State("")
    user_memory = gr.State({})
    user_question = gr.State("")
    memory = gr.State(memory)
    local_memory_qa = gr.State(local_memory_qa)
    data_args = gr.State(data_args)


    # memory_dir = gr.State("")
    user_memory_index = gr.State(None)
    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")
    gr.Markdown(description_top)
    with gr.Row(scale=1).style(equal_height=True):
        input_component = gr.Textbox(lines=1, show_label=False,placeholder="在这里输入你的名字...")
        output_component = gr.Textbox(show_label=False)
        with gr.Column(min_width=30, scale=0.3):
            NameBtn = gr.Button("Remember Me")
    with gr.Row(scale=1).style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row(scale=1):
                chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")
            with gr.Row(scale=1):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Enter text"
                    ).style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("Send")
                with gr.Column(min_width=70, scale=1):
                    cancelBtn = gr.Button("Stop")
            

            with gr.Row(scale=1):
                emptyBtn = gr.Button(
                    "🧹 New Conversation",
                )
                retryBtn = gr.Button("🔄 Regenerate")
                delLastBtn = gr.Button("🗑️ Remove Last Turn")
            with gr.Row(scale=1).style(equal_height=True):
                UpdateMemoryBtn = gr.Button("🔄 Summarize Memory Bank")
            
        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=512,
                        value=512,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
    
    
    gr.Markdown(description)
    enter_name_args = dict(
        fn=enter_name,
        inputs=[input_component,memory,local_memory_qa,data_args],
        outputs=[output_component,user_memory,memory,user_name,user_memory_index],
    ) 
    update_memory_args = dict(
        fn=summarize_memory_event_personality,
        inputs=[data_args,memory,user_name],
        outputs=[user_memory],
        show_progress=True
    )
    predict_args = dict(
        fn=predict_new,
        inputs=[
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
            user_name,
            user_memory,
            memory,
            user_memory_index,
            local_memory_qa
        ], 
        outputs=[chatbot, history, memory, status_display],
        show_progress=True,
    )
    retry_args = dict(
        fn=retry,
        inputs=[
            user_input,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
            user_name,
            user_memory_index,
            local_memory_qa
        ],
        outputs=[chatbot, history, memory, status_display],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])
    # Name
    NameBtn.click(**enter_name_args)
    # Chatbot
    cancelBtn.click(cancel_outputing, [], [status_display])
    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_question, user_input, submitBtn, cancelBtn],
        show_progress=True,
    )
    
    user_input.submit(**transfer_input_args).then(**predict_args)

    submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        inputs=[history],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    retryBtn.click(**retry_args)

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )
    UpdateMemoryBtn.click(**update_memory_args)
demo.title = "SiliconFriend"

if __name__ == "__main__":
    reload_javascript()
    # 启动界面，并使其始终在外部窗口中运行
    # interface.launch(share=True)
    print('Starting demo with share=True')
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
        share=True
    )
 