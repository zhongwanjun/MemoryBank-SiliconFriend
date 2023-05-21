# -*- coding:utf-8 -*-
import os, shutil
import logging
import sys, openai
import copy
import time, platform

# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type
import signal,json
import gradio as gr
import nltk
import torch
from langchain.llms import AzureOpenAI,OpenAIChat


prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(prompt_path)
from utils.sys_args import data_args,model_args
from utils.app_modules.utils import *
#  
from utils.app_modules.presets import *
from utils.app_modules.overwrites import *
from utils.prompt_utils import *
from utils.memory_utils import enter_name_llamaindex, summarize_memory_event_personality, save_local_memory
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

VECTOR_SEARCH_TOP_K = 2

api_path ='api_key_list.txt'

def read_apis(api_path):
    api_keys = []
    with open(api_path,'r',encoding='utf8') as f:
         lines = f.readlines()
         for line in lines:
             line = line.strip()
             if line:
                 api_keys.append(line)
    return api_keys


memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
if not os.path.exists(memory_dir):
    json.dump({},open(memory_dir,"w",encoding="utf-8"))

global memory 
memory = json.load(open(memory_dir,"r",encoding="utf-8"))
language = 'en'
user_keyword = generate_user_keyword()[language]
ai_keyword = generate_ai_keyword()[language]
boot_name = boot_name_dict[language]
boot_actual_name = boot_actual_name_dict[language]
meta_prompt = generate_meta_prompt_dict_chatgpt()[language]
new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatgpt()[language]
api_keys = read_apis(api_path)

deactivated_keys = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

def chatgpt_chat(prompt,system,history,gpt_config,api_index=0):
        retry_times,count = 5,0
        response = None
        while response is None and count<retry_times:
            try:
                request = copy.deepcopy(gpt_config)
                # print(prompt)
                if data_args.language=='en':
                    message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "Hi!"},
                    {"role": "assistant", "content": f"Hi! I'm {boot_actual_name}! I will give you warm companion!"}]
                else:
                     message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "你好！"},
                    {"role": "assistant", "content": f"你好，我是{boot_actual_name}！我会给你温暖的陪伴！"}]
                for query, response in history:
                    message.append({"role": "user", "content": query})
                    message.append({"role": "assistant", "content": response})
                message.append({"role":"user","content": f"{prompt}"})
                # print(request)
                # print(message)
                response = openai.ChatCompletion.create(
                    **request, messages=message)
                # print(prompt)
            except Exception as e:
                print(e)
                if 'This key is associated with a deactivated account' in str(e):
                    deactivated_keys.append(api_keys[api_index])
                api_index = api_index+1 if api_index<len(api_keys)-1 else 0
                while api_keys[api_index] in deactivated_keys:
                    api_index = api_index+1 if api_index<len(api_keys)-1 else 0
                openai.api_key = api_keys[api_index]

                count+=1
        if response:
            response = response['choices'][0]['message']['content'] #[response['choices'][i]['text'] for i in range(len(response['choices']))]
        else:
            response = ''
        return response



def predict_new(
    text,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    user_name,
    user_memory,
    user_memory_index,
    service_context,
    api_index
):
    if text == "":
        return history, history, "Empty context."
    system_prompt,related_memo = build_prompt_with_search_memory_llamaindex(history,text,user_memory,user_name,user_memory_index,service_context=service_context,api_keys=api_keys,api_index=api_index,meta_prompt=meta_prompt,new_user_meta_prompt=new_user_meta_prompt,data_args=data_args,boot_actual_name=boot_actual_name)
    chatgpt_config = {"model": "gpt-3.5-turbo",
        "temperature": temperature,
        "max_tokens": max_length_tokens,
        "top_p": top_p,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2, 
        'n':1
        }
    
    if len(history) > data_args.max_history:
        history = history[data_args.max_history:]
    # print(history)
    response = chatgpt_chat(prompt=text,system=system_prompt,history=history,gpt_config=chatgpt_config,api_index=api_index)
    result = response

   
    torch.cuda.empty_cache()
   
    a, b = [[y[0], y[1]] for y in history] + [
                    [text, result]], history + [[text, result]]
    # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] ,history 
    if user_name:
        save_local_memory(memory,b,user_name,data_args)
    
    return a, b, "Generating..."
     
    

def main(): 
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm_predictor = LLMPredictor(llm=OpenAIChat(model_name="gpt-3.5-turbo"))
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    history = []
    global stop_stream
    print('Please Enter Your Name:')
    user_name = input("\nUser Name：")
    if user_name in memory.keys():
        if input('Welcome back. Would you like to summarize your memory? If yes, please enter "yes"') == "yes":
            user_memory = summarize_memory_event_personality(data_args, memory, user_name)
    hello_msg,user_memory,user_memory_index = enter_name_llamaindex(user_name,memory,data_args)
    print(hello_msg)
    api_index = 0
    print("Welcome to use SiliconFriend model，please enter your question to start conversation，enter \"clear\" to clear conversation ，enter \"stop\" to stop program")
    while True:
        query = input(f"\n{user_name}：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("Welcome to use SiliconFriend model，please enter your question to start conversation，enter \"clear\" to clear conversation ，enter \"stop\" to stop program")
            continue
        count = 0
        history_state, history, msg = predict_new(text=query,history=history,top_p=0.95,temperature=1,max_length_tokens=1024,max_context_length_tokens=200,
                                                  user_name=user_name,user_memory=user_memory,user_memory_index=user_memory_index,
                                                  service_context=service_context,api_index=api_index)
        if stop_stream:
                stop_stream = False
                break
        else:
            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(output_prompt(history_state,user_name,boot_actual_name), flush=True)
                signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)       
        print(output_prompt(history_state,user_name,boot_actual_name), flush=True)

if __name__ == "__main__":
    main()