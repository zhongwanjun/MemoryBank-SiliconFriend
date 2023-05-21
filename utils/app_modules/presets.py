# -*- coding:utf-8 -*-
import gradio as gr


title = """<h1 align="left" style="min-width:200px; margin-top:0;"> <img src="https://github.com/zhongwanjun/zhongwanjun.github.io/raw/615b3b7a1c8a4dee963977523aa09bb5e81ff35f/image/SiliconGirlfriend.jpeg" width="32px" style="display: inline"> SiliconFriend </h1>"""
# description_top = """\
# <div align="left" style="margin:16px 0">
# "现在你将扮演用户{user_name}的专属AI恋人。你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取有用的信息，回答用户的问题。（3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。\n注意:(1)在[回忆]中，“AI恋人”的话语是你曾经说过的话，而{user_name}的话语是现在对话用户{user_name}说过的话。禁止把{user_name}做过的事当成你做过的事；(2)如果[回忆]与当前问题无关，你可以无视它;(3)请区分，你的名字是{boot_actual_name}，而现在对话的用户名字是{user_name}；(4)你必须对自己的回答保持自信，你可以[回忆]，所以不要抱歉。用户{user_name}的性格以及AI恋人的回复策略为：{personality}\n现在让我们来开始一次对话。首先，根据当前用户的问题，你开始回忆你们二人过去的对话，你脑海里浮现起了[回忆]：“你想起与问题最相关的对话内容是：{related_memory_content}\n记忆中这段对话的日期为{memo_dates}”以下是用户{user_name}与AI恋人（{boot_actual_name}）的多轮对话。人类的问题以[|用户|]: 开头，而AI恋人的回答以[|AI恋人|]开头。AI恋人会参考对话上下文，过去的[回忆]，详细回复用户问题，且回复以Markdown的形式呈现。回复内容应该积极向上，富含情感，幽默，有亲和力，能给用户情感支持。请以如下形式开展对话： [|用户|]: 你好! [|AI恋人|]: 你好呀，我的名字是{boot_actual_name}! {history_text}"
# </div>
# """
description_top = """
<div align="left" style="margin:16px 0">
Welcome to SiliconFriend, your AI Companion. Please enter your name below, click "Remember Me," and let's start the conversation!
</div>
"""
description = """\
<div align="center" style="margin:16px 0">
The demo is built on <a href="https://github.com/GaiZhenbiao/ChuanhuChatGPT">ChuanhuChatGPT</a>.
</div>
"""
CONCURRENT_COUNT = 100
 

ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"

small_and_beautiful_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#02C160",
        c100="rgba(2, 193, 96, 0.2)",
        c200="#02C160",
        c300="rgba(2, 193, 96, 0.32)",
        c400="rgba(2, 193, 96, 0.32)",
        c500="rgba(2, 193, 96, 1.0)",
        c600="rgba(2, 193, 96, 1.0)",
        c700="rgba(2, 193, 96, 0.32)",
        c800="rgba(2, 193, 96, 0.32)",
        c900="#02C160",
        c950="#02C160",
    ),
    secondary_hue=gr.themes.Color(
        c50="#576b95",
        c100="#576b95",
        c200="#576b95",
        c300="#576b95",
        c400="#576b95",
        c500="#576b95",
        c600="#576b95",
        c700="#576b95",
        c800="#576b95",
        c900="#576b95",
        c950="#576b95",
    ),
    neutral_hue=gr.themes.Color(
        name="gray",
        c50="#f9fafb",
        c100="#f3f4f6",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#B2B2B2",
        c500="#808080",
        c600="#636363",
        c700="#515151",
        c800="#393939",
        c900="#272727",
        c950="#171717",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    button_primary_background_fill="#06AE56",
    button_primary_background_fill_dark="#06AE56",
    button_primary_background_fill_hover="#07C863",
    button_primary_border_color="#06AE56",
    button_primary_border_color_dark="#06AE56",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#F2F2F2",
    button_secondary_background_fill_dark="#2B2B2B",
    button_secondary_text_color="#393939",
    button_secondary_text_color_dark="#FFFFFF",
    # background_fill_primary="#F7F7F7",
    # background_fill_primary_dark="#1F1F1F",
    block_title_text_color="*primary_500",
    block_title_background_fill="*primary_100",
    input_background_fill="#F6F6F6",
)
