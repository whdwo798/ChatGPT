import streamlit as st
import os

st.title("환영합니다!")

openai_api_key = st.text_input("OpenAI API Key 입력", type="password")

btn1 = st.button("설정", key="openai_btn")


tavily_api_key = st.text_input("Tavily API Key 입력", type="password", key="tavily")

btn2 = st.button("설정", key="tavily_btn")

if btn1:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.markdown("✅ OpenAI API Key가 설정되었습니다.")

if btn2:
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    st.markdown("✅ Tavily API Key가 설정되었습니다.")