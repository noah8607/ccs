import os
from openai import OpenAI

model = os.getenv("LLM_MODEL", "qwen2.5-32b-instruct")
api_key = os.getenv("LLM_API_KEY", "sk-006ae6b678ee4ae3bc1059861f985cd0")
base_url = os.getenv("LLM_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

client = OpenAI(
    api_key=api_key, 
    base_url=base_url,
)

def analysis(text):

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': '你是一家汽车销售公司的经理，下面是你手下的销售人员和客户的一通电话录音，你要负责分析录音并给销售人员提供工作指导。让我们一步一步来做。首先，分析整个电话沟通过程中客户的情绪变化、客户关注的内容、客户提出的要求、销售人员答应客户的事情，然后根据以上内容推测这个客户最终成单的可能性。分析完成后，给销售人员提供指导意见，应当至少包含以下部分： **电话内容** 用最多50字概括电话的通话内容 **客户关注** 用简要的语言列出最多3条客户最关心的内容，如果提到了具体的型号、价格的重要信息，要记清楚 **后续约定** 简要列出销售和客户在电话中一致同意的后续工作与任务，如果包括时间一定要写清楚 **成单概率** 推测该客户最终会在这个销售这里下单的概率 **工作建议** 简要列出最多三条实际和可操作的建议，让销售人员接下来去操作，来尽量提高客户最终成单的可能性。'},
            {'role': 'user', 'content': text}],
        )
    return(completion.choices[0].message.content)
    

if __name__ == "__main__":

    from data import AllData
    ad = AllData()

    newtrs = []
    for r in AllData.alldata['train_records']:
        r['coach'] = analysis(r['text'])
        newtrs.append(r)
        print(f"已添加{len(newtrs)}/{len(AllData.alldata['train_records'])}")
    AllData.alldata['train_records'] = newtrs
    ad.save()