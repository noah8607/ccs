from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import aiohttp
import io
from pydantic import BaseModel, HttpUrl
import mimetypes
import os

from stt import transcriptions
from llm import analysis
from embed import embedding
from cls import predict

app = FastAPI(title="录音分析 API", description="提供录音文件上传、识别和分析功能")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加m4a的MIME类型
mimetypes.add_type('audio/mp4', '.m4a')

class AnalysisRequest(BaseModel):
    file_url: HttpUrl
    id: str
    need_coach: bool = True
    need_classification: bool = True

class AnalysisResponse(BaseModel):
    id: str
    text: str
    coach: Optional[str] = None
    classification: Optional[dict] = None

def is_supported_audio(content_type: str, filename: str) -> bool:
    """检查是否为支持的音频格式"""
    supported_types = {'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/x-m4a'}
    supported_extensions = {'.mp3', '.m4a'}
    
    # 检查content-type
    if content_type in supported_types:
        return True
    
    # 检查文件扩展名
    ext = os.path.splitext(filename.lower())[1]
    return ext in supported_extensions

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AnalysisRequest):
    """
    分析录音文件
    
    参数:
    - file_url: MP3或M4A文件的URL
    - id: 请求ID
    - need_coach: 是否需要销售指导分析
    - need_classification: 是否需要客户分类
    
    返回:
    - id: 请求ID
    - text: 语音识别文本
    - coach: 销售指导建议（如果请求）
    - classification: 客户分类结果（如果请求）
    """
    try:
        # 下载文件
        async with aiohttp.ClientSession() as session:
            async with session.get(str(request.file_url)) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="无法下载音频文件")
                
                content_type = resp.headers.get('content-type', '')
                if not is_supported_audio(content_type, str(request.file_url)):
                    raise HTTPException(status_code=400, detail="不支持的音频格式，仅支持MP3和M4A格式")
                
                contents = await resp.read()
        
        # 语音识别
        stt_result = transcriptions(contents)
        if 'error' in stt_result:
            raise HTTPException(status_code=400, detail=stt_result['text'])
        
        text = stt_result['text']
        response = {
            "id": request.id,
            "text": text
        }
        
        # 销售指导分析
        if request.need_coach:
            coach_result = analysis(text)
            response["coach"] = coach_result
        
        # 客户分类
        if request.need_classification and request.need_coach:  # 分类需要同时有文本和指导建议
            text_embedding = embedding(text)
            coach_embedding = embedding(coach_result)
            t, p, t5, p5 = predict(list(text_embedding) + list(coach_embedding))
            response["classification"] = {
                "type": t,
                "probability": float(p),
                "top5_types": t5,
                "top5_probabilities": [float(x) for x in p5]
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8501, reload=True) 