## 介绍

MemoryBank是为大型语言模型（LLM）设计的记忆机制。它允许模型访问相关记忆，通过不断更新记忆实现演化，并通过综合过去的互动来适应用户个性。受艾宾浩斯遗忘曲线理论启发，MemoryBank采用了一种模拟人类记忆行为的独特更新机制。这使得AI可以根据记忆的重要性和时间推移，有选择地遗忘或强化记忆，从而打造一个自然的记忆系统。MemoryBank可以轻松地与闭源模型（如[ChatGPT](https://chat.openai.com)）和开源模型（如[ChatGLM](https://github.com/THUDM/ChatGLM-6B) 和 [BELLE](https://github.com/LianjiaTech/BELLE)进行集成。 

![](resources/framework.png)

SilconFriend 是一款集成了MemoryBank的双语AI聊天陪伴机器人。 a bilingual LLM-based chatbot with MemoryBank Mechanism in a long-term AI Companion scenario. 通过在大量心理对话数据进行LoRA微调，SiliconFriend在互动中展现出优秀的共情能力。我们进行了一系列实验，包括对真实用户对话进行的定性分析和通过ChatGPT生成的模拟对话进行的定量分析。 实验结果显示，搭载了MemoryBank的SiliconFriend展示出了出色的长期陪伴能力，它能够提供共情性回应、回忆相关记忆，并理解用户的个性。所有的实验都在Tesla A100 80GB GPU和cuda 11.7环境下完成。SiliconFriend分别提供基于ChatGLM和BWELLE两个版本的[LoRA 模型](https://github.com/zhongwanjun/MemoryBank-SiliconFriend/releases/tag/LoRA_checkpoint)。

![](resources/chat_comparison.png)

## 使用方式

### 环境安装

使用pip安装依赖: `pip install -r requirement.txt`.

### Demo

#### SiliconFriend(ChatGLM) 网页版 Demo

设置[SiliconFriend-ChatGLM/launch.sh](SiliconFriend-ChatGLM/launch.sh)中的API KEY 'OPENAI_API_KEY' 和LoRA模型 'adapter_model'，并运行仓库中的[SiliconFriend-ChatGLM/launch.sh](SiliconFriend-ChatGLM/launch.sh):

```shell
./SiliconFriend-ChatGLM/launch.sh
```

#### SiliconFriend(ChatGLM) 命令行 Demo

设置[SiliconFriend-ChatGLM/launch_cmd.sh](SiliconFriend-ChatGLM/launch_cmd.sh)中的API KEY 'OPENAI_API_KEY' 和LoRA模型 'adapter_model'， 并运行仓库中的[SiliconFriend-ChatGLM/launch_cmd.sh](SiliconFriend-ChatGLM/launch_cmd.sh):

```shell
./SiliconFriend-ChatGLM/launch_cmd.sh
```
#### SiliconFriend(ChatGPT) 网页版 Demo

设置[SiliconFriend-ChatGLM/launch_cmd.sh](SiliconFriend-ChatGLM/launch_cmd.sh)中的API KEY 'OPENAI_API_KEY'， 并运行仓库中的 [SiliconFriend-ChatGPT/launch.sh](SiliconFriend-ChatGPT/launch.sh):

```shell
./SiliconFriend-ChatGPT/launch.sh
```

## Citation

如果你觉得我们的工作有帮助的话，请考虑引用下列论文：

```
@article{
  zhong2023memorybank,
  title={MemoryBank: Enhancing Large Language Models with Long-Term Memory},
  author={Zhong, Wanjun and Guo, Lianghong and Gao, Qiqi and Wang, Yanlin},
  journal={arXiv preprint arXiv:2305.10250},
  year={2023}
}
```