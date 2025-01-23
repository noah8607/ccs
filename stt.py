# -*- coding: utf-8 -*-
import os
import logging
import uvicorn
import ffmpeg
import numpy as np
from typing import BinaryIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


SAMPLE_RATE = 16000

# 模型加载
model_path = os.getenv("MODEL_PATH", "models/SenseVoiceSmall")
#model_path = os.getenv("MODEL_PATH", "models/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn")
vad_path = os.getenv("VAD_PATH", "models/speech_fsmn_vad_zh-cn-16k-common-pytorch")
spk_path = os.getenv("CAM_PATH", "models/speech_eres2net_sv_zh-cn_16k-common")
punc_path = os.getenv("PUNC_PATH", "models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

# 支持任意时长音频输入
vad_enable = os.getenv("VAD_ENABLE", True)

# 推理方式
device_type = os.getenv("DEVICE_TYPE", "cpu")

# 设置用于 CPU 内部操作并行性的线程数
cpu_num = os.getenv("ncpu", 4)

# 语言
language = os.getenv("language", "zh")

batch_size = os.getenv("batch_size", 64)

use_itn = os.getenv("use_itn", True)

if vad_enable:
        # 准确预测
    model = AutoModel(
        model=model_path,
        vad_model=vad_path,
        vad_kwargs={"max_single_segment_time": 30000},
        #spk_model=spk_path,
        #punc_model=punc_path,
        trust_remote_code=False,
        device=device_type,
        ncpu=cpu_num,
        disable_update=True
    )
else:
    # 快速预测
    model = AutoModel(
        model=model_path,
        trust_remote_code=False,
        device=device_type,
        ncpu=cpu_num,
        disable_update=True
    )

print(f"音频模型成功加载到 {device_type}")

def transcriptions(file):

    data = load_audio(file)

    if data is None or len(data) == 0:
        return {"text": "音频文件解码错误", "error": "ErrFFMPEG"}
    res = model.generate(
        input=data,
        cache={},
        language=language,
        use_itn=use_itn,
        merge_vad=True,
        batch_size=batch_size,
    )

    result = rich_transcription_postprocess(res[0]["text"])

    return {"text": result}


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file)
            )

        except ffmpeg.Error as e:
            return None

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

if __name__ == "__main__":

    from data import AllData
    ad = AllData()

    newtrs = []
    for r in AllData.alldata['train_records']:
        mp3 = r['mp3']
        ts = transcriptions(mp3)
        if not 'error' in ts:
            r['text'] = ts['text']
            newtrs.append(r)
        print(f"已添加{len(newtrs)}/{len(AllData.alldata['train_records'])}")
    AllData.alldata['train_records'] = newtrs
    ad.save()

