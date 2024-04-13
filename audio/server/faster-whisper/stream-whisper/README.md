# 使用 Faster-whisper 模拟实时语音转写



# 使用方法
## 服务端


```bash
git clone https://github.com/ultrasev/stream-whisper
apt -y install libcublas11
cd stream-whisper
pip install -r requirements.txt
```

注：
- `libcublas11` 是 NVIDIA CUDA Toolkit 的依赖，如果需要使用 CUDA Toolkit，需要安装。
- 经 [@muzian666](https://github.com/muzian666) 提示，aioredis 包目前仍然不支持 Python3.11，Python 版本建议 3.8 ~ 3.10

