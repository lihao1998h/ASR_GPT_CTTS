import gradio as gr
import os
import json
from offline_ASR_and_translate import *
import os
import sys
sys.path.insert(0, os.getcwd())
    

from knowledge.gpt import gpt

with open('language.json', 'r', encoding='utf-8') as f:
    language_dict = json.load(f)

def process_input(text_input, record):
    input_chinese_name, output_chinese_name = '英语', '英语'
    
    if output_chinese_name in language_dict:
        output_lang = language_dict[output_chinese_name]
    else:
        raise ValueError("output_lang未找到该语言")
    
    try: 
        input_lang = language_dict[input_chinese_name]
    except:
        input_lang = 'auto'

    # 1 ASR

    if text_input and not record:
        print('识别文本')
        ori_text, translated_text = ASR_and_translate(text_input, input_lang, output_lang)
    elif record and not text_input:
        print('识别音频或麦克风输入文件')
        ori_text, translated_text = ASR_and_translate(record, input_lang, output_lang)
    else:
        raise ValueError("输入格式错误")    
    
    # 2 写prompt to GPT
    prompt = f'You are a professional English teacher, please talk to me in English, the following is my input: {translated_text}'
    response = gpt(prompt, model='gpt-3.5-turbo')
    
    
    # 3 TTS
    ref_wav_path = None # 读取某个参考的音频，3-10s
    ref_ori_text, ref_translated_text = ASR_and_translate(ref_wav_path, 'auto', 'en')

    prompt_text = res # 参考音频的文本，如果为空则ref_free=True
    prompt_language = lang
    text = prompt # 要TTS的文本
    text_language = 'en' 

    save_path = 'TEMP/output_wav/test.wav'
    if os.path.exists(save_path):
        os.makedirs(save_path)
    TTS_offline(ref_wav_path, prompt_text, prompt_language, text, text_language, save_path)
    
    
    
    return ori_text, translated_text


# 定义Gradio界面
title = "离线英语语音识别+智能回复+自定义语音输出"
description = """支持三种输入形式:\n 1.上传.wav文件\n 2.输入文本\n 3.点击Record按钮\n
                点击Predict获取翻译结果。\n
                注意不要同时输入文本以及上传音频
                """
inputs = [
    # 目前只做英语
    # gr.Dropdown(choices=list(language_dict.keys()), label="输入语言(为空时自动识别)", value=None),
    # gr.Dropdown(choices=list(language_dict.keys()), label="目标语言", value="中文"),
    
    gr.Textbox(label="直接输入文本"),
    gr.Audio(sources=["upload", "microphone"], type="filepath", label="上传文件/麦克风输入"),
    # gr.inputs.Checkbox(label="开启麦克风监听，按ctrl+r停止", default=False, type="bool")
    # gr.Text(label="URL (YouTube, etc.)/链接(YouTube,其他)"),
]

outputs = [
    gr.Textbox(label="源文本或语音识别结果"),
    gr.Textbox(label="翻译结果"),
]

# 定义处理函数，包括对音频文件的预处理
demo = gr.Interface(
    fn=process_input,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    allow_flagging=False,  # 防止用户标记结果
)

# Queue up the demo

demo.launch(share=False, server_name='127.0.0.1', server_port=7861, quiet=True)
