#from email.mime import image
#from re import L
#from httpx import stream
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import base64
import speech_recognition as sr
#import pyttsx3
from gtts import gTTS
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere.chat_models import ChatCohere
from langchain_ollama import ChatOllama

import asyncio
import nest_asyncio
import threading
import keyboard

nest_asyncio.apply()
# 音声入力（認識）関数
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language="ja-JP")
        except:
            return "" 
        
st.header("Mr.Yas Mic_to_text 🤗")        
image_placeholder = st.sidebar.empty()
already_displayed = False         
while True:
    if not already_displayed:
        print("話しかけてください...")
        st.write("🤗話しかけてください...")
        already_displayed = True
    st.session_state.user_input = ""
    st.session_state.user_input = speech_to_text()
      
    # 対話ループ 

    #ユーザーの音声入力を表示
    with st.chat_message('user'):   
        st.write(st.session_state.user_input) 



