import bs4
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader


# 제목
st.title("네이버 뉴스 요약")


# 사이드바
with st.sidebar:
    naver_news_url = st.text_input("네이버 뉴스 URL 입력")
    summary_btn = st.button("요약")


# 네이버 뉴스 기사를 요약하는 체인을 생성
def create_naver_news_summary_chain():
    prompt = load_prompt("prompts/naver_news.yaml", encoding="utf-8")

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
    )

    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return stuff_chain


# summary_btn 이 눌렸을 때
if summary_btn:
    with st.spinner("네이버의 뉴스 기사를 요약하는 중입니다..."):
        # 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
        loader = WebBaseLoader(
            web_paths=(naver_news_url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    "div",
                    attrs={
                        "class": [
                            "newsct_article _article_body",
                            "media_end_head_title",
                        ]
                    },
                )
            ),
        )

        # 네이버 뉴스 사이트에서 기사의 내용을 크롤링하여 가져옵니다.
        docs = loader.load()

        # 요약을 위한 Chain 을 생성
        news_chain = create_naver_news_summary_chain()

        # 요약된 결과를 받습니다.
        summary_result = news_chain.invoke({"context": docs})

        with st.chat_message("ai"):
            st.markdown(summary_result)
            st.markdown(f"**✅ 원문 링크: {naver_news_url}**")