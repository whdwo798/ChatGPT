import streamlit as st
from langchain_core.messages.chat import ChatMessage
from agent import create_agent

st.title("검색 기반 에이전트")


# 메시지를 저장할 list 를 생성합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Agent 생성
if "agent" not in st.session_state:
    st.session_state.agent = create_agent(k=5)


# 채팅 메시지에 새로운 메시지를 추가하는 함수
def add_message(role, message):
    # 메시지 list 에 새로운 대화(메시지)를 추가합니다.
    st.session_state.messages.append(ChatMessage(role=role, content=message))


# 이전의 대화기록을 모두 출력하는 함수
def print_message():
    for message in st.session_state.messages:
        # 대화를 출력
        st.chat_message(message.role).write(message.content)


# 이전 대화 기록을 모두 출력
print_message()

# 채팅 입력창
user_input = st.chat_input("궁금한 내용을 입력해 주세요.")

# 만약에 유저가 채팅을 입력하면
if user_input:
    st.chat_message("user").write(user_input)

    # 체인 생성
    chain = st.session_state["agent"]

    if chain is not None:

        with st.spinner("AI 검색하여 답변을 준비 중입니다..."):

            # chain 을 실행서 ai_answer 를 받습니다.
            ai_answer = chain.invoke({"input": user_input})

            with st.chat_message("ai"):
                st.markdown(ai_answer["output"])

            # 대화를 추가
            add_message("user", user_input)
            # ai 의 답변을 추가
            add_message("ai", ai_answer["output"])