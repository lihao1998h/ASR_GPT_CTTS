import asyncio
import logging
import websockets
import time
from collections import deque
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any


async def asyncformer(sync_func: Callable, *args, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, sync_func, *args, **kwargs)


CONVERSION_QUEUE = asyncio.Queue()
CONVERSATION = deque(maxlen=100)
MODEL_SIZE = "/data/lh/ai_cybertransformation/audio/models/faster-whisper-large-v3"
CN_PROMPT = '' # '聊一下基于faster-whisper的实时/低延迟语音转写服务'
logging.basicConfig(level=logging.INFO)
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
logging.info('Model loaded')

async def handle_audio(websocket):
    def b_transcribe():
        # transcribe audio to text
        start_time = time.time()
        segments, info = model.transcribe("temp/audio/chunk.mp3",
                                          beam_size=5,
                                          initial_prompt=CN_PROMPT)
        end_time = time.time()
        period = end_time - start_time
        text = ''
        for segment in segments:
            t = segment.text
            if t.strip().replace('.', ''):
                text += ', ' + t if text else t
        return text, period
    
    while True:
        try:
            chunk = await websocket.recv()
            with open('temp/audio/chunk.mp3', 'wb') as f:
                f.write(chunk)
                
            text, period = await asyncformer(b_transcribe)
            t = text.strip().replace('.', '')
            logging.info(t)
            CONVERSATION.append(text)
            
            
            while len(CONVERSATION) != 0:
                ret = CONVERSATION.pop()
                # 将转写结果发送回客户端
                await websocket.send(ret)

        except websockets.exceptions.ConnectionClosedError:
            break


# async def b_transcribe(chunk):
#     # 将音频数据保存到临时文件
#     with open('temp/audio/chunk.mp3', 'wb') as f:
#         f.write(chunk)

#     # 转写音频到文本
#     start_time = time.time()
#     segments, info = model.transcribe("temp/audio/chunk.mp3", beam_size=5, initial_prompt=CN_PROMPT)
#     end_time = time.time()
#     period = end_time - start_time
#     text = ''
#     for segment in segments:
#         t = segment.text
#         if t.strip().replace('.', ''):
#             text += ', ' + t if text else t
#     return text, period


async def transcribe_server() -> None:
    async with websockets.serve(handle_audio, "localhost", 8765):
        logging.info(f"WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever until cancelled


async def main():
    await asyncio.gather(transcribe_server())


if __name__ == '__main__':
    asyncio.run(main())