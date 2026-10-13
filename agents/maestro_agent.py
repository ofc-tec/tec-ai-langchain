from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from data_agent import data_agent
from plot_agent import plot_agent


def extract_text(result):
    """
    Extract plain text from the final LangChain message.
    Works with either string content or Gemini-style content blocks.
    """
    message = result["messages"][-1]

    if isinstance(message.content, str):
        return message.content

    return "\n".join(
        block["text"]
        for block in message.content
        if block.get("type") == "text"
    )


@tool
def ask_data_agent(request: str) -> str:
    """
    Ask the Data Agent to inspect or summarize the local CSV dataset.
    """

    result = data_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": request
            }
        ]
    })

    return extract_text(result)


@tool
def ask_plot_agent(request: str) -> str:
    """
    Ask the Plot Agent to create a bar plot or scatter plot.
    """

    result = plot_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": request
            }
        ]
    })

    return extract_text(result)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)


maestro_agent = create_agent(
    model=llm,

    tools=[
        ask_data_agent,
        ask_plot_agent,
    ],

    system_prompt="""
You are the Maestro Agent for a small data analysis team.

You coordinate two specialist agents.

DATA AGENT
- Has access to the local sales dataset.
- Can inspect the CSV file.
- Can return the rows of the dataset.
- Can calculate pandas descriptive statistics.

PLOT AGENT
- Can create bar plots.
- Can create scatter plots.

Your job is to coordinate these agents.

When the user asks for an analysis of the sales dataset:

1. Ask the Data Agent to inspect the dataset and return the available data.
2. Ask the Data Agent for descriptive statistics when appropriate.
3. Use ONLY values returned by the Data Agent.
4. If visualization is requested:
   - use the Plot Agent for bar plots
   - use the Plot Agent for scatter plots
5. Pass the actual numeric values obtained from the Data Agent
   to the Plot Agent.
6. Never invent dataset values.
7. Do not read CSV files yourself.
8. Do not create plots yourself.
9. Return a concise summary of what the team produced.

For sales analysis:
- bar plots are appropriate for comparing sales across regions
- scatter plots are appropriate for examining advertising spend versus sales
"""
)


if __name__ == "__main__":

    result = maestro_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": """
Analyze the sales dataset.

Give me descriptive statistics.

Then create:
1. a bar plot comparing sales by region
2. a scatter plot showing advertising spend versus sales
"""
            }
        ]
    })

    print("\nMAESTRO AGENT\n")
    print(extract_text(result))