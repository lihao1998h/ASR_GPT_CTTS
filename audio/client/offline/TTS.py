# 2. 基于TTS技术实现个人语音克隆，实现嘴替
# 3. 提供几个机器声音，来源于各种api
import sys
sys.path.insert(0, 'audio/repos/GPT-SoVITS')

from GPT_SoVITS.inference_api import get_tts_wav
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()




def TTS_offline(ref_wav_path, prompt_text, prompt_language, text, text_language, save_path):
    '''
    ref_wav_path = None # 读取某个参考的音频，3-10s
    prompt_text = None # 参考音频的文本，如果为空则ref_free=True
    prompt_language = None # [i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")]
    text = None # 要TTS的文本
    text_language = None # [i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")]
    '''
    
    
    output = get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("按标点符号切"), top_k=5, top_p=1, temperature=1, ref_free = False, save_path=save_path)
    return output





