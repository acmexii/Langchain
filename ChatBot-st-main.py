import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

load_dotenv()

def create_agent_chain(history):
    chat = ChatOpenAI(
        model_name=os.getenv("OPENAI_API_MODEL"),
        temperature=os.getenv("OPENAI_API_TEMPERATURE")
    )
    tools = load_tools(["ddg-search", "wikipedia"])
    prompt = hub.pull("hwchase17/openai-functions-agent")
    memory = ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True
    )
    agent = create_openai_tools_agent(chat, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory)

st.title("Langchain-streamlit-app")
history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).markdown(message.content)
    
prompt = st.chat_input("Enter your message")
print(prompt)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = create_agent_chain(history)        
        response = agent_chain.invoke(
            {"input" : prompt},
            {"callbacks" : [callback]})
        
        st.markdown(response["output"])