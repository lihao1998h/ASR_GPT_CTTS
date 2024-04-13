import os 
# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"

os.environ['OPENAI_API_KEY'] = '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'

import openai
# openai.api_base = "https://openai.wndbac.cn/v1"
from openai import OpenAI
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
# print(client.files.list())
# print(client.models.list())


model_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-vision-preview", 'gpt-4-0125-preview']

input_messages = [{"role": "system", "content": "You are a helpful assistant."}]

def gpt(prompt, model="gpt-3.5-turbo"):
    global input_messages
    append_message = {"role": "user", "content": prompt}

    input_messages.append(append_message)
    response = client.chat.completions.create(
        model=model,
        messages=input_messages
    )
    #遇到connection error问题，看https://blog.csdn.net/Oooops_/article/details/134811558

    output_content = response.choices[0].message.content

    input_messages.append({"role": "assistant", "content": output_content}) # 计划加入多轮对话

    # print(f'input tokens:', {response["usage"]["prompt_tokens"]}) # type: ignore
    # print(f'prompt tokens: {response["usage"]["completion_tokens"]}') # type: ignore
    # print(f'total tokens: {response["usage"]["total_tokens"]}') # type: ignore
    
    return output_content # type: ignore

if __name__ == "__main__":
    prompt = '''
    
    我现在的研究内容是这样写的：

在现代计算机视觉领域中，图像语义识别对于智能系统的理解与决策具有重要意义。然而，当前的研究面临着一系列挑战：（1）实际图像语义识别任务极其复杂，各类影响因素相互交织、高度耦合，构建反映真实因果关系的图模型颇具难度；（2）现有方法普遍依赖于预训练知识，而先验知识本身可能成为一个混淆变量，导致样本特征与类别标签间存在虚假关联性。这种伪相关性若在训练和测试数据分布之间不一致时，将严重影响模型的泛化能力和鲁棒性；（3）在多数图像语义识别场景中，混杂因子往往是复杂且难以直接观测的，如何有效地消除这些混杂效应成为亟待解决的关键问题。

针对上述难点，本项目提出以下创新解决方案：（1）基于对比学习与迭代分区不变风险最小化的特征语义解耦方法：通过对比不同环境下的图像特征表现，系统性地分离并剔除混杂因子对图像语义识别的影响，实现特征与语义的深度解耦，从而提升模型在多变环境中的稳定性和准确性。（2）基于后门调整与遗忘学习的去偏小样本学习策略：利用因果推理中的后门调整原理，精准识别并校正由混杂因子引起的伪相关性，结合遗忘学习机制，在有限的小样本数据集上动态优化模型参数，减少先验知识带来的潜在偏差。（3）基于前门调整与原型补全的增量小样本语义分割技术：提出了因果干预模块来消除导致语义偏移的因果效应，其中逐步和自适应地更新旧类原型并以前门调整方式减轻混淆偏见，还提出了一个原型补全模块，通过从情景学习中迁移的知识引导，融合新类样本和旧类原型的特征，从而补全新类缺失的语义特征。

这三个步骤依次递进，紧密结合，共同构成了一个完整的基于因果推断的图像语义混杂因子控制框架。该框架有望突破现有图像语义识别技术的瓶颈，为提高图像识别系统的可靠性和泛化能力提供新的理论指导与实践方案，从而有力推动图像语义识别领域的深入发展。

但是现在需要改表达结构，注意：强制参考下面参考资料的结构，根据每一句从上面提取信息，而且你还需要增加一段，因为我有三个方法，参考资料只有两个，要求，不要出现参考资料的内容，学术化的表达，以下是参考资料：

注意：强制按照参考资料的结构，而不是我原本的结构。

软体机器人感知技术一直以来都是该领域的难点问题，这一挑战在交互环境控制任务中进一步被放大——由于被动柔顺机构受力形变的特点，使得软体机械臂极有可能在外力作用下局部产生较大的不可预测形变，这对内传感器的抵抗变形和集中应力的能力提出更高的要求。

基于光纤的内感知方法为软体机器人状态 反馈问题提供了解决方案，近年来得到了研究者们广泛关注。但传统光纤因两端 固定的布置方式和材质较高脆性，在机器人受到外界交互影响而局部弯曲和扭转 程度较大的情况下极易发生断裂；且当变形超出应变-波长(光强）呈线性的区间 时，测量精度严重下降，因此在交互作业中存在局限性。针对环境交互导致的局 部形变程度、集中应力急剧提升的情况，本项目首先研究大变形下适用的基于光纤的姿态、力感知方法，拟采用内嵌光纤光栅（FBG）传感器，设计螺旋型光纤构型，提升对软体机械臂径向伸缩、弯曲应变的耐受力；拟设计多维力的空间解耦装置检测末端三维接触力大小，为机械臂末端操作控制提供必要力反馈信息。

为了保证作业任务的安全性、操作的精准性，交互控制任务中除了必要的自身形状、姿态、位置反馈外，对交互状态（交互存在性判断、交互位置、交互力 大小等）的感知也是必不可少的。但当前技术大都要求交互类型的先验信息，降低不确定交互任务中适用性；或是需要对各类型信息分步求解，对实时性提出挑 战。针对交互环境控制任务中对多类型交互状态的实时反馈需求，并考虑交互类 型难以提前预判问题，项目拟研究融合记忆机制的交互状态检测方法，采用预训练的长短期记忆（LSTM）网络同步实现交互存在性判断、交互位置和交互力大小测量。该网络检测结果为控制任务提供必要反馈，且结合内容一中理论模型框 架可以实时更新约束模型，保证后续算法设计的可行性。
    '''
    print(gpt(prompt, model='gpt-3.5-turbo'))