#from email.mime import image
#from re import L
#from httpx import stream
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import base64
import speech_recognition as sr
import pyttsx3

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
#from torch import res

r = sr.Recognizer()
engine = pyttsx3.init()
nest_asyncio.apply()

def init_page():
    st.set_page_config(
        page_title="My Chat",
        page_icon="🤗"
    )
    st.header("My Chat 🤗")
    st.sidebar.title("Options")

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]    

def select_model():
    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = st.sidebar.slider(
        "Temperature(回答バラツキ度合):", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    models = ("llava-llama3","llava","llava-phi3", "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro",
              "command-r-plus","mistral","llama3.1","aya","ELYZA-JP:8b")
    model = st.sidebar.radio("Choose a model（大規模言語モデルを選択）:", models)

    if model == "llava-llama3":
        st.session_state.model_name = "llava-llama3"
        return ChatOllama(
            temperature=temperature,
            #model_name=st.session_state.model_name,
            model=st.session_state.model_name
        )   #.bind(images=[base64_image])
    elif model == "llava":
        st.session_state.model_name = "llava"
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name
        ) 
    elif model == "llava-phi3":
        st.session_state.model_name = "llava-phi3"
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name
        )      
    elif model == "GPT-4o":  #"gpt-4o 'gpt-4o-2024-08-06'" 有料？、Best
        st.session_state.model_name = "gpt-4o"
        return ChatOpenAI(
            temperature=temperature,
            model=st.session_state.model_name,
            max_tokens=512,  #指定しないと短い回答になったり、途切れたりする。
            streaming=True,
        )
    elif model == "Claude 3.5 Sonnet": #コードがGood！！
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            temperature=temperature,
            #model=st.session_state.model_name,
            model_name=st.session_state.model_name,  
            max_tokens_to_sample=2048,  
            timeout=None,  
            max_retries=2,
            stop=None,  
        )
    elif model == "Gemini 1.5 Pro":
        st.session_state.model_name = "gemini-1.5-pro-latest"
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=st.session_state.model_name
        )
    elif model == "command-r-plus":
        st.session_state.model_name = "command-r-plus"
        return ChatCohere(
            temperature=temperature,
            model=st.session_state.model_name,
            streaming=True,
        )
    elif model == "mistral":
        st.session_state.model_name = "mistral"
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name
        )
    elif model == "llama3.2":
        st.session_state.model_name = "llama3.2"
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name
        )
    elif model == "aya":
        st.session_state.model_name = "aya"
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name
        )
    elif model == "ELYZA-JP:8b":
        st.session_state.model_name = "ELYZA-JP:8b"
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name
        )
    
#######################################################################
#  LLM問答関数   
async def query_llm(user_input,frame):
    print("user_input=",user_input)

    try:
            
        # 画像を適切な形式に変換（例：base64エンコードなど）
        # 画像をエンコード
        encoded_image = cv2.imencode('.jpg', frame)[1]
        # 画像をBase64に変換
        base64_image = base64.b64encode(encoded_image).decode('utf-8')  
        #image = f"data:image/jpeg;base64,{base64_image}"
        
        if st.session_state.model_name ==  "keep_gpt-4o":
            llm = st.session_state.llm  
            stream = llm.stream([
                    *st.session_state.message_history,
                    (
                        "user",
                        [
                            {
                                "type": "text",
                                "text": user_input
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "auto"
                                },
                            }
                        ]
                    )
                ])
            #response = chain.invoke(user_input)
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response = st.write_stream(stream) 
            #full = next(stream)
            #for chunk in stream:
                #full += chunk
            #response = full    
            print(response)
          
        
        if st.session_state.model_name ==  "command-r-plus":
            print("st.session_state.model_name=",st.session_state.model_name)
            print(user_input)
            prompt = ChatPromptTemplate.from_messages(
                [
                    *st.session_state.message_history,
                     #("user", f"{user_input}:{base64_image}"),  #やっぱりだめ
                     ("user", f"{user_input}")
                ]
            )
            
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            #stream = chain.stream(user_input,base64_image)
            
            stream = chain.stream({"user_input":user_input,"base64_image": base64_image})
            print("stream=",stream)
            #response = chain.invoke(user_input)
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response =st.write_stream(stream) 
            print("response=",response)

        elif st.session_state.model_name ==  "keep_command-r-plus":
            print("st.session_state.model_name=",st.session_state.model_name)
            prompt = ChatPromptTemplate.from_messages([
                    *st.session_state.message_history,
                    ("user", "{user_input}")  # ここにあとでユーザーの入力が入る
                ])
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            stream = chain.stream(user_input)
            
            #response = chain.invoke(user_input)
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response =st.write_stream(stream) 
            print("response=",response)
        else:
            print("st.session_state.model_name=",st.session_state.model_name)
            prompt = ChatPromptTemplate.from_messages(
                [
                    *st.session_state.message_history,
                    (
                        "user",
                        [
                            {
                                "type": "text",
                                "text": user_input
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            }
                        ],
                    ),
                ]
            )
             
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            #stream = chain.stream(user_input,base64_image)
            stream = chain.stream({"user_input":user_input,"base64_image": base64_image})
            #print("stream=",stream)
            #response = chain.invoke(user_input)
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response =st.write_stream(stream) 
            #print("response=",response)            
        

        print(f"{st.session_state.model_name}=",response)
        

        # 音声出力処理                
        if st.session_state.output_method == "音声":
            speak_thread = speak_async(response)
            # 必要に応じて音声合成の完了を待つ
            speak_thread.join()    
            print("音声再生が完了しました。次の処理を実行します。")
        if engine._inLoop:
            print("音声出力がLOOPになっています。")
            engine.endLoop()
            print("音声再生LOOPを解除しました。次の処理を実行できます")

        # チャット履歴に追加
        st.session_state.message_history.append(("user", user_input))
        st.session_state.message_history.append(("ai", response))
    
        return response
    except StopIteration:
        # StopIterationの処理
        print("StopIterationが発生")
        pass

    user_input = ""
    base64_image = ""
    frame = ""    
#######################################################################

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):   
        self.frame = frame.to_ndarray(format="bgr24")
        return frame
#######################################################################
# 音声入力（認識）関数
def speech_to_text():
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language="ja-JP")
        except:
            return ""
#######################################################################
#音声出力関数
def speak_async(text):
    def run():
        engine.say(text)
        engine.startLoop(False)
        engine.iterate()
        engine.endLoop()
    thread = threading.Thread(target=run)
    thread.start()
    return thread
####################################################################### 
####################################################################### 
#def main():
async def main():     
    ###################################################################    
    #画面表示
    #streamlit.errors.StreamlitAPIException: `set_page_config()` can only be called once per app page, 
    # and must be called as the first Streamlit command in your script
    init_page()
    init_messages()
    # 音声認識の設定
    #recognizer = sr.Recognizer()
    # 音声合成の設定
    #engine = pyttsx3.init()

    #stで使う変数初期設定
    st.session_state.llm = select_model()
    st.session_state.input_method = ""
    st.session_state.user_input = ""
    st.session_state.result = ""
    st.session_state.frame = "" 
    
    col, col2 = st.sidebar.columns(2)
     # 各列にボタンを配置
    with col:
        # 入力方法の選択
        input_method = st.sidebar.radio("入力方法", ("テキスト", "音声"))
        st.session_state.input_method = input_method
     # 各列にボタンを配置
    with col2:
        # 出力方法の選択
        output_method = st.sidebar.radio("出力方法", ("テキスト", "音声"))
        st.session_state.output_method = output_method

    # チャット履歴の表示 (第2章から少し位置が変更になっているので注意)
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)
    
    # サイドバーにWebRTCストリームを表示
    with st.sidebar:
        st.header("Webcam Stream")
        webrtc_ctx=webrtc_streamer(
                key="speech-to-text",
                desired_playing_state=True, 
                mode=WebRtcMode.SENDRECV,
                #audio_receiver_size=1024,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                #audio_processor_factory=AudioTransformer,
                video_processor_factory=VideoTransformer,
            )   
                 
    
    ###################################################################        
    ###############################################################################
    #入力データ設定
    #キャプチャー画像入力
    if webrtc_ctx.video_transformer:  
        frame = webrtc_ctx.video_transformer.frame  #VideoProcessor.frame 

    # テキスト入力フォーム
    if st.session_state.input_method == "テキスト":
        button_input = ""
        # 4つの列を作成
        col1, col2, col3, col4 = st.columns(4)

        # 各列にボタンを配置
        with col1:
            if st.button("画像の内容を説明して"):
                button_input = "画像の内容を説明して"

        with col2:
            if st.button("前の画像と何が変わりましたか？"):
                button_input = "前の画像と何が変わりましたか？"

        with col3:
            if st.button("石川県の観光地を教えてください。"):
                button_input = "石川県の観光地を教えてください。"

        with col4:
            if st.button("CIDPとは？"):
                button_input = "CIDPとは？"

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            if st.button("日本語に翻訳してください。"):
                button_input = "日本語に翻訳してください。"


        with col6:
            if st.button("善悪は何で決まりますか？"):
                button_input = "善悪は何で決まりますか？"

        with col7:
            if st.button("日本の観光地を教えてください。"):
                button_input = "日本の観光地を教えてください。"

        with col8:
            if st.button("今日の料理はなにがいいかな"):
                button_input = "今日の料理はなにがいいかな"

        #query="""以下のチャットボットアプリのpythonコードを教えてください。
                #1.StreamlitでWeb表示する。
                #2.可能な限りlangchainライブラリを使用してください。
                #3.マルチボーダル対応のLLMを5モデル（gpt-4o,claude3.5 sonnet,gemini-1.5-pro,command-r-plus,mistral)から選択して使用するボタンを設定する。
                #4.streamlit_webrtcを使用し、常時Webカメラ画像を表示し、カメラ画像について問い合わせもできるようにする。
                #5.LLMへの問い合わせ入力をキーボードによるテキスト入力の場合と音声入力の場合を選択して使用するボタンを設定する。
                #6.LLMへ問い合わせ入力する際は、Webカメラのキャプチャ画像も入力する。
                #7.LLMからの出力をテキスト出力のみする場合とテキスト出力と音声出力の両方出力する場合とを選択して使用するボタンを設定する。
                #8.音声出力、LLMへの問い合わせ応答など非同期処理にする必要があるかもしれません。
                #streamlit_webrtを使用し、LLMに音声、画像、テキストを入力してテキストと音声で出力するチャットボットのコードを教えて"""
        #if st.button(query):
                #button_input = query
          
        if button_input !="":
            st.session_state.user_input=button_input
        
        # クリアボタンが押された場合、セッションステートの値をクリア
        #Streamlitのst.text_input()で以前入力したテキストをクリアするには、key引数を動的に変更する方法が効果的です。
        #ボタンをクリックするたびにtext_inputのkeyが変更され、入力フィールドがリセットされます。
        # これは、Streamlitの再実行の仕組みを利用して入力をクリアする効果的な方法です。
        #if 'text_input' not in st.session_state:
            #st.session_state.text_input = random.randint(1, 100000)
        #if st.button("前のテキストをクリア"):
            #clear_text()
        
        text_input =st.chat_input("テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
        #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
        if text_input:
            st.session_state.user_input=text_input
            text_input=""

        if st.session_state.user_input:
            print("user_input=",st.session_state.user_input)
            with st.chat_message('user'):   
                st.write(st.session_state.user_input) 
        # 対話ループ 
        # 画像と問い合わせ入力があったときの処理
            if frame is not None and st.session_state.user_input !="":
                st.sidebar.header("Capture Image")
                st.sidebar.image(frame, channels="BGR")
                # if st.button("Query LLM : 画像の内容を説明して"):
                with st.spinner("Querying LLM..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    st.session_state.result= ""
                    result = loop.run_until_complete(query_llm(st.session_state.user_input,frame))
                    st.session_state.result = result
                    result = ""
                    #result = await query_llm(text,frame)
                    st.session_state.user_input=""

    ###############################################################################
    #音声入力（テキストに変換した入力）の対話ループ
    #print("Before_st.session_state.input_method=",st.session_state.input_method)
    if st.session_state.input_method == "音声": 
        already_displayed = False
        st.sidebar.header("Capture Image") 
        image_placeholder = st.sidebar.empty()
         
        while True:
            if not already_displayed:
                print("話しかけてください...")
                st.write("🤗話しかけてください...")
                already_displayed = True
            st.session_state.user_input = ""
            st.session_state.user_input = speech_to_text()
            if keyboard.is_pressed('1') :st.session_state.user_input ="こんばんは"
            if keyboard.is_pressed('2') :st.session_state.user_input ="画像の内容を説明して"
            if keyboard.is_pressed('3') :st.session_state.user_input ="石川県小松市の観光地は？"
            if keyboard.is_pressed('4') :st.session_state.user_input ="有名な道の駅は？"
            if keyboard.is_pressed('5') :st.session_state.user_input ="CIDPとは？"
            if keyboard.is_pressed('6') :st.session_state.user_input ="きょうの料理はなにがいいかな"
            if keyboard.is_pressed('7') :st.session_state.user_input ="宇宙人はいますか？"
            if keyboard.is_pressed('8') :st.session_state.user_input ="私の名前は誠です。"
            if keyboard.is_pressed('9') :st.session_state.user_input ="私の名前は？"
            if keyboard.is_pressed('9') :st.session_state.user_input ="善悪は何で決まりますか？"
            if keyboard.is_pressed('esc') :
                print("音声での問い合わせを終了しました。")
                with st.chat_message('assistant'):   
                    st.write("音声での問い合わせを終了しました。") 
                #break   
            # 対話ループ 
            # 画像と問い合わせ入力があったときの処理
            if webrtc_ctx.video_transformer: #VideoProcessor
                frame = webrtc_ctx.video_transformer.frame  #VideoProcessor.frame 
            if frame is not None and st.session_state.user_input !="":
                #サイドバーに画像を表示
                image_placeholder.image(frame, channels="BGR")
                #ユーザーの音声入力を表示
                with st.chat_message('user'):   
                    st.write(st.session_state.user_input) 
                #LMMの回答を表示 
                with st.spinner("Querying LLM..."):
                    #loop = asyncio.new_event_loop()
                    #asyncio.set_event_loop(loop)
                    #st.session_state.result= ""
                    #result = loop.run_until_complete(query_llm(st.session_state.user_input,frame))
                    result = await query_llm(st.session_state.user_input,frame)
                st.session_state.result = result
                result = ""
                st.session_state.user_input=""
                already_displayed = False
                    
    ###############################################################################  
    ###############################################################################
    #print("after_st.session_state.input_method=",st.session_state.input_method)

if __name__ == "__main__":
    #main()
    #asyncio.run(main())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
