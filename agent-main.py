import os
from dotenv import load_dotenv
load_dotenv()

import langchain

langchain.debug = True
langchain.verbose = False

from langchain.agents import load_tools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.hub import LangChainHub
from langchain.prompts import load_prompt

# 'hub' 인스턴스를 생성합니다.
hub = LangChainHub()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = load_tools(["ddg-search"])
#prompt = hub.load("hwchase17/openai-functions-agent")
prompt = load_prompt("hwchase17/openai-functions-agent")


agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


result = agent_executor.invoke({"input": "서울과 부산의 날씨를 알려줘"})
