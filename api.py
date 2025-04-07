from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import aiohttp
import io
from pydantic import BaseModel, HttpUrl
import mimetypes
import os
import re

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
    call_summary: Optional[str] = None
    customer_focus: Optional[list[str]] = None
    follow_up: Optional[list[str]] = None
    success_rate: Optional[str] = None
    suggestions: Optional[list[str]] = None
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

def parse_coach_content(coach_text: str) -> dict:
    """解析教练反馈内容，提取结构化信息"""
    result = {
        "call_summary": None,
        "customer_focus": [],
        "follow_up": [],
        "success_rate": None,
        "suggestions": []
    }
    
    if not coach_text:
        print("coach_text is empty")
        return result

    # 使用更严格的分割方式
    import re
    
    # 按 ### 分割文本
    sections = re.split(r'###\s*([^#\n]+)', coach_text)
    
    # 移除空字符串
    sections = [s.strip() for s in sections if s.strip()]
    
    # 创建标题到内容的映射
    content_map = {}
    for i in range(0, len(sections)-1, 2):
        title = sections[i].strip()
        content = sections[i+1].strip() if i+1 < len(sections) else ""
        content_map[title] = content
    
    # 提取各个部分的内容
    if "电话内容" in content_map:
        result["call_summary"] = content_map["电话内容"]
    
    if "客户关注" in content_map:
        # 使用更智能的列表项识别
        items = re.findall(r'(?:^|\n)(?:\d+\.|[\-\*•])?\s*(.+?)(?=(?:\n(?:\d+\.|[\-\*•])|\n\n|$))', content_map["客户关注"])
        result["customer_focus"] = [item.strip() for item in items if item.strip()]
    
    if "后续约定" in content_map:
        items = re.findall(r'(?:^|\n)(?:\d+\.|[\-\*•])?\s*(.+?)(?=(?:\n(?:\d+\.|[\-\*•])|\n\n|$))', content_map["后续约定"])
        result["follow_up"] = [item.strip() for item in items if item.strip()]
    
    if "成单概率" in content_map:
        result["success_rate"] = content_map["成单概率"]
    
    if "工作建议" in content_map:
        items = re.findall(r'(?:^|\n)(?:\d+\.|[\-\*•])?\s*(.+?)(?=(?:\n(?:\d+\.|[\-\*•])|\n\n|$))', content_map["工作建议"])
        result["suggestions"] = [item.strip() for item in items if item.strip()]
    
    return result

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
    - call_summary: 电话内容（如果请求）
    - customer_focus: 客户关注点（如果请求）
    - follow_up: 后续约定（如果请求）
    - success_rate: 成单概率（如果请求）
    - suggestions: 工作建议（如果请求）
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
            # 解析结构化数据
            parsed_coach = parse_coach_content(coach_result)
            response.update(parsed_coach)
        
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