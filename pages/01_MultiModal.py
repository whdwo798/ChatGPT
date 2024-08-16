import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal
from langchain_core.messages.chat import ChatMessage

st.title("이미지 인식 GPT")

# 메시지를 저장할 list 를 생성합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []


# 채팅 메시지에 새로운 메시지를 추가하는 함수
def add_message(role, message):
    # 메시지 list 에 새로운 대화(메시지)를 추가합니다.
    st.session_state.messages.append(ChatMessage(role=role, content=message))


# 이전의 대화기록을 모두 출력하는 함수
def print_message():
    for message in st.session_state.messages:
        # 대화를 출력
        st.chat_message(message.role).write(message.content)


with st.sidebar:
    url = st.text_input("이미지 URL을 입력해주세요.")

    system_prompt = st.text_area(
        "시스템 프롬프트",
        """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 
당신의 임무는 주어진 재무제표를 바탕으로 투자에 참고할 만한 유용한 정보를 정리해 주는 것입니다.""",
    )

if url:
    st.image(url)


def create_chain(system_prompt, user_prompt):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o",  # 모델명
    )

    # system_prompt = """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다.
    # 당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""

    # user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

    # 멀티모달 객체 생성
    chain = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)
    return chain


# 이전 대화 기록을 모두 출력
print_message()

# 채팅 입력창
user_input = st.chat_input("궁금한 내용을 입력해 주세요.")

# 만약에 유저가 채팅을 입력하면
if user_input:
    st.chat_message("user").write(user_input)

    # 체인 생성
    chain = create_chain(system_prompt, user_input)

    # chain 을 실행서 ai_answer 를 받습니다.
    answer = chain.stream(url)

    with st.chat_message("ai"):
        # 빈 공간을 만듬
        chat_container = st.empty()

        # ai 의 답변을 출력
        ai_answer = ""

        # 스트리밍 출력
        for token in answer:
            ai_answer += token.content
            chat_container.markdown(ai_answer)

    # 대화를 추가
    add_message("user", user_input)
    # ai 의 답변을 추가
    add_message("ai", ai_answer)