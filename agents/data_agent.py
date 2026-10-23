from pathlib import Path
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool

#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama




CSV_PATH = Path(__file__).parent / "data" / "sales.csv"



@tool
def inspect_csv() -> str:
    """Inspect the local sales dataset and return its rows and column information."""

    df = pd.read_csv(CSV_PATH)

    return f"""
Rows: {len(df)}

Columns and data types:
{df.dtypes.to_string()}

Data:
{df.to_string(index=False)}
"""


@tool
def summarize_csv() -> str:
    """Return pandas descriptive statistics for the local sales dataset."""

    df = pd.read_csv(CSV_PATH)

    return df.describe().to_string()


#llm = ChatGoogleGenerativeAI(
#    model="gemini-3.8-flash",
#    temperature=0,
#)
llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)

data_agent = create_agent(
    model=llm,
    tools=[
        inspect_csv,
        summarize_csv,
    ],
    system_prompt="""
You are a data analysis specialist.

You have access to one local sales dataset.

Use inspect_csv when you need to see the dataset structure or values.
Use summarize_csv when descriptive statistics are requested.

Do not ask for a filename.
Do not invent values.
Use only information returned by your tools.
"""
)


if __name__ == "__main__":

    result = data_agent.invoke({
        "messages": [{
            "role": "user",
            "content": """
Inspect the sales dataset and then give me descriptive statistics.
"""
        }]
    })

    print(result["messages"][-1].content)