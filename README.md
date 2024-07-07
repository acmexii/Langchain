# 대화기록과 Pinecone 벡터DB 검색으로 답변하는 Chatbot (w/ streamlit)
- history : StreamlitChatMessageHistory()
- Vectorstore : Pinecone
- embeddings = OpenAIEmbeddings
- Document Loader : UnstructuredPDFLoader
#### Code
```
index_name = os.environ["PINECONE_INDEX"]
embeddings = OpenAIEmbeddings()
return Pinecone.from_existing_index(index_name, embeddings)
```

### Chain #1 : LangChain의 create_history_aware_retriever를 사용해 과거의 대화 기록을 고려해 질문을 다시 표현하는 Chain 생성
```
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
```

### Chain #2 : 문맥을 고려하여 질문에 답하는 Chain 생성   
```
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
```

### 두 Chain을 연결한 Chain 생성  
```
conversational_retrieval_chain = (
    RunnablePassthrough.assign(context=rephrase_chain | format_docs) | qa_chain
)
```

### in Local
- Local에서 streamlit 실행
```
e.g.> streamlit run Rag-st-main.py --server.port 8080
```
