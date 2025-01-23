# AIO Demo by AlfredG
FROM python:3.11.7

ENV MODE="DEMO"
ENV LANG="C.UTF-8"
ENV TZ="Asia/Shanghai"

RUN apt-get update \
 && apt-get install -y nano dumb-init ffmpeg\
 && rm -rf /var/lib/apt/lists/*

RUN echo "Install AI related packages" \
  && pip3 install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
  && pip3 install --no-cache-dir openai==1.54.4 transformers==4.46.2 numpy==1.26.3 pandas==2.2.0 torchvision==0.20.1 torchaudio==2.5.1 
RUN echo "Install AI models" \
  && pip3 install --no-cache-dir modelscope==1.19.2 funasr==1.1.14 ffmpeg==1.4 ffmpeg-python==0.2.0 fastapi==0.115.4 python-multipart==0.0.17  \
  && pip install vllm==0.6.4.post1 accelerate==1.1.1
RUN echo "Install packages for Streamlit ui" \
  && pip3 install --no-cache-dir streamlit==1.30.0 
RUN echo "Install function packages" \
  && pip3 install --no-cache-dir openpyxl==3.1.2 xlsxwriter==3.1.9
RUN echo "Install commandline management packages" \
  && pip3 install --no-cache-dir tqdm==4.66.1 fire==0.5.0 

# copy source files
COPY ./*.py /app/
COPY ./models /app/models
COPY ./data /app/data

STOPSIGNAL SIGINT
WORKDIR /app
EXPOSE 8888

ENTRYPOINT ["streamlit", "run", "demo.py", "--server.port=8888", "--server.address=0.0.0.0"]

