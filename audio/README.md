# 仅管理server代码

# 支持的服务
1. 语音识别与翻译：基于ASR和翻译技术实现外文转化为母语

- 中文实时语音识别服务 [funasr](#funasr服务端)
- 外文实时语音识别服务 [whisper-live](#funasr服务端)




# 环境准备
```bash
# linux环境下

# 安装ffmpeg
sudo apt update
sudo apt install ffmpeg

# python环境
conda create --name ai_cybertransformation python=3.10

# 安装依赖
pip install faster-whisper
pip install openai-whisper
pip install funasr==1.0.17
pip install modelscope
pip install rotary_embedding_torch
pip install langdetect
pip install transformers
pip install nltk
sudo apt-get install build-essential
sudo apt-get install libportaudio2 portaudio19-dev
pip install pyaudio
pip install websockets
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')" # 可能要开代理


mkdir repos
cd repos
## 安装GPT-SoVITS
git clone https://github.com/RVC-Boss/GPT-SoVITS
pip install -r GPT-SoVITS/requirements.txt

# 模型下载
# 手动下载到models/目录下
# https://huggingface.co/Systran/faster-whisper-large-v3/tree/main

# pip install torch torchvision torchaudio --upgrade -f https://download.pytorch.org/whl/cu101/torch_stable.html



```




# <a id="funasr服务端"></a>FunASR服务端配置与运行指南

参考 
https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md

`cd audio/server`

### 启动中文服务 10095端口
```bash
mkdir online-cpu-zh
cd online-cpu-zh

sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.9
mkdir -p ./funasr-runtime-resources/models

sudo docker run -p 10095:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.9
```

### 进入 docker 后 执行
```bash
cd FunASR/runtime
nohup bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx  \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --port 10095 \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt > log.txt 2>&1 &
```



#  <a id="faster-whisper 服务端"></a>faster-whisper 服务端配置与运行指南

## whisper-live（成功）
- 参考 https://github.com/collabora/WhisperLive

`nohup python audio/server/WhisperLive/run_server.py > audio/server/whisperlive_log.txt 2>&1 &`

## ~~stream-whisper(失败)~~
~~参考 https://github.com/ultrasev/stream-whisper~~

~~whisper-stream 客户端只能在linux运行~~

~~stream-whisper 必须用Redis ，我用websocket修改失败~~


