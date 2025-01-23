import os
# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from transformers import AutoModel

model_name = os.getenv("EMBED_MODEL_PATH", "models/jina-embeddings-v3")

model = AutoModel.from_pretrained(model_name, local_files_only=False, trust_remote_code=True, cache_dir="models/HF")

print(f"词嵌入模型加载完成")

def embedding(text):
    return model.encode(text)

if __name__ == "__main__":

    from data import AllData
    ad = AllData()

    newtrs = []
    for r in AllData.alldata['train_records']:
        e1 = embedding(r['text'])
        e2 = embedding(r['coach'])
        r['embed'] = list(e1)+list(e2)
        newtrs.append(r)
        print(f"已添加{len(newtrs)}/{len(AllData.alldata['train_records'])}")
    AllData.alldata['train_records'] = newtrs
    ad.save()