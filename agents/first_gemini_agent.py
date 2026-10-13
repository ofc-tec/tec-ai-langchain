from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

tools = [add_numbers, multiply_numbers]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="Use tools whenever arithmetic is required."
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is (12 + 8) * 3?"
        }
    ]
})

print(result["messages"][-1].content)