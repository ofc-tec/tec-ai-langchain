from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

web_search = TavilySearch(
    max_results=5
)

agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt="""
    You are a research assistant.

    Use web search whenever the user asks for:
    - current information
    - recent events
    - facts that may have changed
    - information you are unsure about

    Summarize the results clearly and mention the sources used.
    """
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the chance of rain today in Mexico City? Remember the results for the day"
        }
    ]
})


print(result["messages"][-1].content[0]["text"])


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Do you remember the last answer you gave me?"
        }
    ]
})


print(result["messages"][-1].content[0]["text"])