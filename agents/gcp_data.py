
"""GCP Data Agent — LangChain + Google BigQuery MCP."""

import asyncio
import subprocess
import sys

from dotenv import load_dotenv
from fastmcp.client.transports import StreamableHttpTransport
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

PROJECT_ID = "ia-2006"
MCP_URL = "https://bigquery.googleapis.com/mcp"


async def run_gcp_data(question: str) -> str:
    
    # Authenticate using the existing gcloud login
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        text=True,
    ).strip()
    ####################GOOGLE PROVIDED
    #########################################Create protocol for remote comm
    transport = StreamableHttpTransport(
        url=MCP_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
        },
    )
    # Connect, discover tools, create agent, invoke, close connection
    #LAngchain MCPAdapter handles the connection and session management for you

    async with MCPAdapter(transport) as adapter:
        tools = await adapter.list_tools()

        agent = create_agent(
            model=ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                temperature=0,
            ),
            # model=ChatOllama(model="llama3.1:latest", temperature=0),
            tools=tools,
            system_prompt=f"""
            You are a BigQuery Data Analyst for project {PROJECT_ID}.
            Use MCP tools to inspect datasets, schemas, and analyze data.
            Never invent datasets, columns, or results.
            Include executed SQL when relevant.
            """,
                            )

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })

        return result["messages"][-1].content
    async with MCPAdapter(transport) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(
            model=ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                temperature=0,
            ),
            # model=ChatOllama(model="llama3.1:latest", temperature=0),
            tools=tools,
            system_prompt=f"""
            You are a BigQuery Data Analyst for project {PROJECT_ID}.
            Use MCP tools to inspect datasets, schemas, and analyze data.
            Never invent datasets, columns, or results.
            Include executed SQL when relevant.
            """,
                            )

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })

        return result["messages"][-1].content


@tool
def ask_gcp_data(request: str) -> str:
    """Ask the GCP Data Agent to analyze BigQuery."""
    return asyncio.run(run_gcp_data(request))


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "List my datasets."
    print(asyncio.run(run_gcp_data(question)))
