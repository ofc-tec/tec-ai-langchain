from langchain.agents import create_agent
from langchain_core.tools import tool
## MUST PAY AND IMPORT OPEN AI TO USE THIS AGENT
## A free ollama version is provided in first_free_agent.py
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

tools = [add_numbers, multiply_numbers]

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=tools,
    system_prompt="""
    You are a simple math assistant.
    Use tools whenever arithmetic is required.
    """
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