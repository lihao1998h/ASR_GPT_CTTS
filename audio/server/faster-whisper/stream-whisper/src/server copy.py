import asyncio
import logging
import time
from collections import deque

# import aioredis
from faster_whisper import WhisperModel
import websockets

# from .config import REDIS_SERVER
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any


async def asyncformer(sync_func: Callable, *args, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, sync_func, *args, **kwargs)

# CONVERSATION = deque(maxlen=100)
MODEL_SIZE = "/data/lh/ai_cybertransformation/audio/models/faster-whisper-large-v3"
CN_PROMPT = '聊一下基于faster-whisper的实时/低延迟语音转写服务'
logging.basicConfig(level=logging.INFO)
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
logging.info('Model loaded')

CONVERSION_QUEUE = asyncio.Queue()

async def b_transcribe(audio_data):
    # 将音频数据保存到临时文件
    with open('chunk.mp3', 'wb') as f:
        f.write(audio_data)

    # 转写音频到文本
    start_time = time.time()
    segments, info = model.transcribe("chunk.mp3", beam_size=5, initial_prompt=CN_PROMPT)
    end_time = time.time()
    period = end_time - start_time
    text = ''
    for segment in segments:
        t = segment.text
        if t.strip().replace('.', ''):
            text += ', ' + t if text else t
    return text, period

async def handle_audio(websocket, path):
    async for message in websocket:
        # 当接收到新的音频数据时，将其放入队列等待处理
        await CONVERSATION_QUEUE.put(message)

        # 检查是否有待处理的音频数据
        while not CONVERSATION_QUEUE.empty():
            audio_data = await CONVERSATION_QUEUE.get()
            text, period = await b_transcribe(audio_data)

            # 将转写结果发送回客户端
            await websocket.send(text)


async def transcribe_websocket_server():
    async with websockets.serve(handle_audio, "localhost", 8765):
        print("WebSocket server is listening on port 8765")

async def main():
    # 启动WebSocket服务器
    await asyncio.create_task(transcribe_websocket_server())

    # 保持主协程运行
    while True:
        await asyncio.sleep(1)  # 实际应用中可能不需要此循环，此处仅为了保持程序运行
    
    


if __name__ == '__main__':
    asyncio.run(main())
