"""GCP Maestro = original Maestro + BigQuery MCP-backed Data Agent.

Put gcp_maestro.py and gcp_data.py beside the existing plot_agent.py
in ~/tec_agentic_ai/agents; launch from the repo root:
    uv run python agents/gcp_maestro.py
Requires gcloud auth login and an existing GEMINI_API_KEY / GOOGLE_API_KEY.
"""
import asyncio
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from fastmcp.client.transports import StreamableHttpTransport
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from gcp_data import build_gcp_data_agent
from plot_agent import plot_agent

PROJECT_ID = "ia-2006"
MCP_URL = "https://bigquery.googleapis.com/mcp"


def extract_text(result):
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def access_token():
    """Use existing gcloud login; never print the temporary token."""
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()


async def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    # A short-lived token is sufficient for this one-run classroom demo.
    transport = StreamableHttpTransport(
        url=MCP_URL,
        headers={
            "Authorization": f"Bearer {access_token()}",
            "x-goog-user-project": PROJECT_ID,
        },
    )
    async with MCPAdapter(transport) as adapter:
        mcp_tools = await adapter.list_tools()
        print("BigQuery MCP tools:", ", ".join(t.name for t in mcp_tools))
        data_agent = build_gcp_data_agent(mcp_tools)

        @tool
        async def ask_gcp_data(request: str) -> str:
            """Ask GCP Data to discover BigQuery metadata or execute read-only SQL analytics."""
            response = await data_agent.ainvoke({
                "messages": [{"role": "user", "content": request}]
            })
            return extract_text(response)

        @tool
        async def ask_plot_agent(request: str) -> str:
            """Ask the existing Plot Agent to draw a bar or scatter chart from numeric values."""
            # Plot Agent currently uses synchronous LangChain execution.
            response = await asyncio.to_thread(
                plot_agent.invoke,
                {"messages": [{"role": "user", "content": request}]},
            )
            return extract_text(response)

        maestro = create_agent(
            #model=ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0),
            model=ChatOllama(
                model="llama3.1:latest",
                temperature=0
                                ),
            tools=[ask_gcp_data, ask_plot_agent],
            checkpointer=InMemorySaver(),
            system_prompt="""You are GCP Maestro. Coordinate GCP Data and Plot Agent.
GCP Data has BigQuery MCP access. Plot Agent can draw bar and scatter plots.
For a data question, ask GCP Data for verified SQL aggregate results first.
Pass only the actual numerical rows and labels returned by GCP Data to Plot
Agent; NEVER invent or estimate values. Do not execute SQL or create plots
yourself. Keep transfers small. Confirm plot filenames in your final reply.
If GCP Data cannot access the dataset or SQL tool, report the error; do not
pretend that analysis was performed.""",
        )

        prompt = os.getenv(
            "GCP_MAESTRO_QUESTION",
            "Using project ia-2006, inspect the BTS flights table schema and "
            "compute average arrival delay for the 10 airlines with the most "
            "flights. Ask Plot Agent for a bar chart of those exact averages. "
            "Explain the results and include the SQL used.",
        )
        result = await maestro.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "gcp-maestro-demo"}},
        )
        print("\n--- GCP MAESTRO ---\n" + extract_text(result))


if __name__ == "__main__":
    asyncio.run(main())
