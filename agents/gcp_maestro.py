"""GCP Maestro = original Maestro + BigQuery MCP-backed Data Agent.
Put gcp_maestro.py and gcp_data.py beside the existing plot_agent.py
in ~/tec_agentic_ai/agents; launch from the repo root:
    uv run python agents/gcp_maestro.py
Requires gcloud auth login and an existing GEMINI_API_KEY / GOOGLE_API_KEY.
"""
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from gcp_data import ask_gcp_data
from plot_agent import plot_agent


@tool
def ask_plot_agent(request: str) -> str:
    """
    Delegate chart creation to the Plot Agent.

    Provide verified numerical values and descriptive labels.
    The Plot Agent creates bar or scatter plots using Matplotlib.
    """
    result = plot_agent.invoke({
        "messages": [
            {"role": "user", "content": request}
        ]
    })
    return extract_text(result)




def extract_text(result):
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )

def build_gcp_maestro():
    """Build the shared orchestrator for CLI and MCP consumers."""
    return create_agent(
        # Keep exactly one model active; switch imports above with the model.
        model=ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0),
        #model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite"),
        #model=ChatOllama(model="llama3.1:latest",temperature=0),
        tools=[ask_gcp_data, ask_plot_agent],
        checkpointer=InMemorySaver(),
        system_prompt="""You are GCP Maestro.Your main tool is GCP Data, which can analyze BigQuery datasets and return verified SQL results. 
                GCP Data handles BigQuery analysis.
                For a data question, ask GCP Data for verified SQL aggregate results first.
                Agent; NEVER invent or estimate values. Do not execute SQL or create plots
                yourself. Keep transfers small. Confirm plot filenames in your final reply.
                If GCP Data cannot access the dataset or SQL tool, report the error; do not
                NEVER  try to delete or modify datasets or tables.respond with 'That kind of operation must be ahutorized bydata owner' 
                """,
                    )

def main():
    maestro = build_gcp_maestro()
    prompt = os.getenv(
        "GCP_MAESTRO_QUESTION",
        "Using project ia-2006, inspect the BTS flights table schema and "
        "compute average arrival delay for the 10 airlines with the most "
        "flights. Ask Plot Agent for a bar chart of those exact averages. "
        "Explain the results and include the SQL used.",
    )
    result = maestro.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "gcp-maestro-demo"}},
    )
    print("\n--- GCP MAESTRO ---\n" + extract_text(result))


if __name__ == "__main__":
    main()
