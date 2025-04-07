# 汽车销售通话分析系统
FROM ubuntu:22.04

# 设置环境变量
ENV LANG="C.UTF-8" \
    TZ="Asia/Shanghai" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="models/SenseVoiceSmall" \
    LLM_MODEL="qwen2.5-32b-instruct" \
    LLM_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL="https://mirrors.cloud.tencent.com/pypi/simple/" \
    PIP_TRUSTED_HOST="mirrors.cloud.tencent.com"

# 配置apt源为腾讯云并安装基础依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.cloud.tencent.com/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        python3.11-venv \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# 创建工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://mirrors.cloud.tencent.com/pypi/simple/

# 复制应用代码和模型
COPY ./*.py ./
COPY ./models ./models
COPY ./data ./data

# 暴露API端口
EXPOSE 8501

# 使用uvicorn启动FastAPI应用
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8501"]
