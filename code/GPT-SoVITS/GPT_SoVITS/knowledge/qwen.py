import dashscope
dashscope.api_key="sk-be7bf2ca2f074844a8538f6878aadb79"

from http import HTTPStatus


def qwen(prompt_text, model='qwen-turbo'):
    # model: ['qwen-turbo']
    '''
    https://help.aliyun.com/zh/dashscope/developer-reference/api-details?spm=a2c4g.11186623.0.i0
    模型名称

模型简介

模型输入/输出限制

qwen-turbo

通义千问超大规模语言模型，支持中文、英文等不同语言输入。

模型支持8k tokens上下文，为了保证正常的使用和输出，API限定用户输入为6k tokens。

qwen-plus

通义千问超大规模语言模型增强版，支持中文、英文等不同语言输入。

模型支持32k tokens上下文，为了保证正常的使用和输出，API限定用户输入为30k tokens。

qwen-max

通义千问千亿级别超大规模语言模型，支持中文、英文等不同语言输入。随着模型的升级，qwen-max将滚动更新升级，如果希望使用稳定版本，请使用qwen-max-1201。

模型支持8k tokens上下文，为了保证正常的使用和输出，API限定用户输入为6k tokens。

qwen-max-1201

通义千问千亿级别超大规模语言模型，支持中文、英文等不同语言输入。该模型为qwen-max的快照稳定版本，预期维护到下个快照版本发布时间（待定）后一个月。

模型支持8k tokens上下文，为了保证正常的使用和输出，API限定用户输入为6k tokens。

qwen-max-longcontext

通义千问千亿级别超大规模语言模型，支持中文、英文等不同语言输入。

模型支持30k tokens上下文，为了保证正常的使用和输出，API限定用户输入为28k tokens。
    '''
    resp = dashscope.Generation.call(
        model=model,
        prompt=prompt_text
    )
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if resp.status_code == HTTPStatus.OK:
        # print(resp.output)  # The output text
        # print(resp.usage)  # The usage information
        return resp.output.text
    else:
        print(resp.code)  # The error code.
        print(resp.message)  # The error message.
    
    return resp.output.text


# res = qwen('用萝卜、土豆、茄子做饭，给我个菜谱。')
 