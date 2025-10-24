from typing import List, Union
from langchain.tools import tool
from langchain_core.agents import AgentAction, AgentFinish
from dotenv import load_dotenv
from langchain_core.tools import Tool, render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
load_dotenv()


@ tool 
def get_text_length(text: str) -> int:

    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text}")
    text = text.strip("'\n").strip(   '"'  )  # stripping away non alphabetic characters just in case
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str ):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found ")




if __name__ == "__main__":


    print("Hello ReAct LangChain!")
    #print (get_text_length.invoke(input={'text':'Cat ate a mouse'}))
    
    tools = [get_text_length]
    template = template = """

    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action should be only the value of the variable avoid things like text=
    Observation: the result of the action

    ... (this Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer
    if you used a tool dont include final answer in the output.
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}

    """

    prompt = PromptTemplate.from_template(template=template).partial(tools= render_text_description(tools) , tool_names= ", ".join( [t.name for t in tools]))
    #llm = ChatOpenAI(temperature=0 , stop= ['\nObservation'])
    llm = ChatOllama(temperature=0, model="llama3.1:latest").bind(stop=['\nObservation'])
    intermediate_steps =[]
    agent =  {'input': lambda x:x['input'], 'agent_scratchpad': lambda x: format_log_to_str (x['agent_scratchpad'])}   | prompt | llm  | ReActSingleInputOutputParser()# LCEL  LAngchain expresion lenguage 
    #agent_step: Union [AgentAction, AgentFinish] = agent.invoke({"input":"what is the length in characters of the word Cat?"  , "agent_scratchpad":intermediate_steps})
    agent_step = agent.invoke({"input":"what is the length in characters of the word Cat?"  , "agent_scratchpad":intermediate_steps})
    #print (agent_step)
    if isinstance(agent_step,AgentAction):
        tool_name= agent_step.tool
        tool_to_use= find_tool_by_name(tools,tool_name)
        tool_input= agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))

        print (f'observation recieved from tool call-> {observation}')
        intermediate_steps.append((agent_step, str(observation)))
    print (intermediate_steps)
    agent_step: Union [AgentAction, AgentFinish] = agent.invoke({"input":"what is the length in characters of the word 'Cat'? " , "agent_scratchpad":intermediate_steps})
    
    print (f'agent_step 2 {agent_step}')
    if isinstance(agent_step,AgentFinish):
        print ( f'final answer {agent_step.return_values}')
    