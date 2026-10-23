"""BigQuery-backed Data Agent with authentication and MCP session management.

Run:
    uv run python agents/gcp_data.py "List the datasets I can access."

Requires gcloud auth login and GEMINI_API_KEY / GOOGLE_API_KEY in the
environment or the repository .env file. Consumes Google's BigQuery MCP
server; this module does not expose an MCP server.
"""
import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastmcp.client.transports import StreamableHttpTransport
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_ollama import ChatOllama

load_dotenv()

PROJECT_ID = "ia-2006"
MCP_URL = "https://bigquery.googleapis.com/mcp"


def _access_token():
    """Use existing gcloud login; never print the temporary token."""
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()


@asynccontextmanager
async def open_gcp_data_agent():
    """Authenticate and yield an agent with a live, automatically closed session.

    Requires an existing gcloud login and GEMINI_API_KEY / GOOGLE_API_KEY.
    The short-lived access token is intended for a single classroom demo run.
    """
    token = await asyncio.to_thread(_access_token)
    transport = StreamableHttpTransport(
        url=MCP_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
        },
    )
    async with MCPAdapter(transport) as adapter:
        mcp_tools = await adapter.list_tools()
        yield build_gcp_data_agent(mcp_tools)


def build_gcp_data_agent(mcp_tools):
    """Give all discovered MCP tools to the specialist LangChain agent."""
    return create_agent(
        # Keep exactly one model active; switch imports above with the model.
        model=ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0),
        #model=ChatOllama(model="llama3.1:latest", temperature=0),
        tools=mcp_tools,
        system_prompt=f"""You are GCP Data, a careful analytical specialist for project {PROJECT_ID}.
Use BigQuery MCP tools to inspect actual dataset/table schemas before writing SQL.
For analysis, push aggregations (AVG, COUNT,
GROUP BY, APPROX_QUANTILES, CORR) to BigQuery. Return small, accurate
structured JSON-compatible results with labels, units, and values.
LIMIT result sizes sensibly (e.g. 30-100 rows for charts).
Never request unbounded raw tables or make up numbers or schemas.
Quote the executed SQL and flag unavailable data or
permission failures. Use only project {PROJECT_ID}.""",
    )


async def _run_gcp_data(request: str) -> str:
    async with open_gcp_data_agent() as agent:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": request}]
        })
        content = result["messages"][-1].content
        if isinstance(content, str):
            return content
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )


def run_gcp_data(request: str) -> str:
    """Run a standalone request, managing authentication and session cleanup."""
    return asyncio.run(_run_gcp_data(request))


@tool
def ask_gcp_data(request: str) -> str:
    """Ask GCP Data to work with BigQuery using its available MCP tools."""
    return run_gcp_data(request)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        f"List the datasets I can access in project {PROJECT_ID}. "
        "Briefly describe up to 5 datasets using their metadata. "
    )
    print(run_gcp_data(question))
