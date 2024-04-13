import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将父目录路径添加到sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

sys.path.insert(0, 'audio/repos/GPT-SoVITS')
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

from audio.client.offline.offline_ASR_and_translate import *
from audio.client.offline.TTS import TTS_offline
# from knowledge.gpt import gpt


# 1 输入：将用户麦克风的输入保存为wav



# 2 wav转文字
# wav_path = '/data/lh/ai_cybertransformation/temp/audio/en.wav'
# res, lang = ASR(wav_path)

res = 'Hello, Im fien. thank u.'

# 3 写prompt to GPT
prompt = f'You are a professional English teacher, Please score my English and tell me if I have any language problems, the following is my input: {res}. Dont appear redundant output, just dialogue.'
# response = gpt(prompt, model='gpt-3.5-turbo')
response = prompt

# 4 TTS
ref_wav_path = '/data/lh/ai_cybertransformation/temp/audio/vocal_Default_20240302-135717.wav_10.wav_0000232960_0000480960.wav' # 请上传3~10秒内参考音频，超过会报错！
res, lang = ASR(ref_wav_path)

prompt_text = res # 参考音频的文本，展示ASR结果，支持手动更改，越准确效果越好
prompt_language = i18n("中文") # [i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")]

text = response # 要TTS的文本
text_language = i18n("英文")

save_path = 'TEMP/output_wav/vocal_Default_20240302-135717.wav_10.wav_0000048000_0000232960.wav'
if os.path.exists(save_path):
    os.makedirs(save_path)
out_path = TTS_offline(ref_wav_path, prompt_text, prompt_language, text, text_language, save_path)


# 5 播放wav

