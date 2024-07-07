import os
import logging
from add_document import initialize_vectorstore
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title("Langchain-streamlit-RAG-app")
history = StreamlitChatMessageHistory()

# Vectorstore와 retriever 생성
vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever()
    
for message in history.messages:
    st.chat_message(message.type).markdown(message.content)

# New Prompt 입력
prompt = st.chat_input("Enter your message")
logger.info(prompt)

# Chain #1 : LangChain의 create_history_aware_retriever를 사용해,
# 과거의 대화 기록을 고려해 질문을 다시 표현하는 Chain 생성
rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "위의 대화에서, 대화와 관련된 정보를 찾기 위한 검색 쿼리를 생성해 주세요."),
    ]
)
rephrase_llm = ChatOpenAI(
    model_name=os.environ["OPENAI_API_MODEL"],
    temperature=os.environ["OPENAI_API_TEMPERATURE"],
)
rephrase_chain = create_history_aware_retriever(
    rephrase_llm, retriever, rephrase_prompt
)

# Chain #2 : 문맥을 고려하여 질문에 답하는 Chain 생성    
callback = StreamlitCallbackHandler(st.container())
qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "아래의 문맥만을 고려하여 질문에 답하세요.\n\n{context}"),
            (MessagesPlaceholder(variable_name="chat_history")),
            ("user", "{input}"),
        ]
    )

qa_llm = ChatOpenAI(
    model_name=os.environ["OPENAI_API_MODEL"],
    temperature=os.environ["OPENAI_API_TEMPERATURE"],
    streaming=True,
    # callbacks=[callback],
)

qa_chain = qa_prompt | qa_llm | StrOutputParser()

# 두 Chain을 연결한 Chain 생성  
conversational_retrieval_chain = (
    RunnablePassthrough.assign(context=rephrase_chain | format_docs) | qa_chain
)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        ai_message = conversational_retrieval_chain.invoke(
            {"input": prompt, "chat_history": history.messages},
            #{"callbacks" : [callback]}
        )
                
        st.markdown(ai_message)  

    # 두 Chain을 연결한 Chain 생성  
    history.add_user_message(prompt)
    history.add_ai_message(ai_message)
