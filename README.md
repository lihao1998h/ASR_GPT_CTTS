ASR_GPT_CTTS
语音识别+GPT+自定义文本转语音

# 项目目录
1. code中包含程序代码
2. 一键启动.bat
3. 

模型训练可以在kaggle或者colab中进行，本地仅推理，大约仅需要4G显存即可


多种音色可以对话，可以是预设的，也可以是自己训练的


todo：音频预处理和模型训练单独分离出来




## 功能（待修改）：

1. **零样本文本到语音（TTS）：** 输入 5 秒的声音样本，即刻体验文本到语音转换。

2. **少样本 TTS：** 仅需 1 分钟的训练数据即可微调模型，提升声音相似度和真实感。

3. **跨语言支持：** 支持与训练数据集不同语言的推理，目前支持英语、日语和中文。

4. **WebUI 工具：** 集成工具包括声音伴奏分离、自动训练集分割、中文自动语音识别(ASR)和文本标注，协助初学者创建训练数据集和 GPT/SoVITS 模型。

**查看我们的介绍视频 [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw)**



1v1 模仿老师声音 陪练（audio-text-gpt（英文对话）-text-TTS（定制化*）-audio）


线上 直播授课 （互动如何保证？）

录课：课件音频自动生成：只有课件（课件老师做） 有老师声音  

# 非开发者看这里

# 开发者看这里


## 环境准备
```bash


# python环境
conda create --name ASR_GPT_CTTS python=3.9 -y

# 安装GPT-SoVITS依赖
pip install -r requirements.txt
conda install ffmpeg -y


# 安装其他依赖
pip install langdetect
pip install dashscope

# pip install openai-whisper
# pip install funasr==1.0.17
# pip install modelscope
# pip install rotary_embedding_torch
# pip install transformers
# pip install nltk
# sudo apt-get install build-essential
# sudo apt-get install libportaudio2 portaudio19-dev
# pip install pyaudio
# pip install websockets
# python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')" # 可能要开代理



# 模型下载
# 手动下载到models/目录下
# https://huggingface.co/Systran/faster-whisper-large-v3/tree/main

# pip install torch torchvision torchaudio --upgrade -f https://download.pytorch.org/whl/cu101/torch_stable.html



```


# 致谢

GPT-SoVITS可以参考[demo video](https://www.bilibili.com/video/BV12g4y1m7Uw)和[github仓库](https://github.com/RVC-Boss/GPT-SoVITS)
