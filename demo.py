from io import StringIO
import streamlit as st

st.set_page_config(page_title="Demo", page_icon="🏠")

from data import AllData
from stt import transcriptions
from llm import analysis
from embed import embedding
from cls import predict

ad = AllData()
types = ad.types

st.markdown("**上传与分析电话录音**")

with st.container(border=True):
    file = st.file_uploader("上传电话录音", type="mp3")

    if st.button("分析"):
        with st.expander("语音识别", expanded=True):
            text = transcriptions(file.read())['text']
            st.write(text)
        with st.expander("分析指导", expanded=True):
            coach = analysis(text)
            st.markdown(coach)
        with st.expander("客户分类", expanded=True):
            t, p, t5, p5 = predict(list(embedding(text))+list(embedding(coach)))

            st.write(t)
            st.write(p)



