from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import  TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda 
from langchain import hub
import os

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse


load_dotenv()

tools = [TavilySearch()]
llm= ChatOpenAI(model="gpt-4")
#llm = ChatOllama(temperature=0, model="llama3.1:latest")
structured_llm = llm.with_structured_output(AgentResponse)

output_parser= PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt = hub.pull("hwchase17/react")

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")

#react_prompt_with_format_instructions = PromptTemplate(
#    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
#    input_variables=["input", "agent_scratchpad", "tool_names"],
#).partial(format_instructions=output_parser.get_format_instructions())



agent = create_react_agent(    ## NOT REALLY AN AGENT SO FAR MORE OF A CHAIN
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions
)
agent_executor= AgentExecutor(agent=agent , tools=tools, verbose = True)

extract_output = RunnableLambda(lambda x: x["output"])
#parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor|extract_output|structured_llm  #parse_output

def main():
    print("Hello from langchain-course!")
    result = chain.invoke(   input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                            }   
    )
    print(result)
    

if __name__ == "__main__":
    main()

