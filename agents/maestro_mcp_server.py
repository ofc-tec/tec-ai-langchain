"""Expose the existing GCP Maestro as one MCP tool for OpenCode.

Copy to ~/tec_agentic_ai/agents/maestro_mcp_server.py
Launch from repository root: uv run python agents/maestro_mcp_server.py

Uses the original gcp_data.py and plot_agent.py unchanged. This process
speaks MCP over stdio: do not print or log to stdout.
"""
from dotenv import load_dotenv

load_dotenv()
import asyncio
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


from gcp_data import build_gcp_data_agent
from plot_agent import plot_agent

PROJECT_ID = "ia-2006"
MCP_URL = "https://bigquery.googleapis.com/mcp"
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

mcp = FastMCP("GCP Maestro")
# The existing Plot Agent writes fixed filenames (bar_plot.png, scatter_plot.png).
# Serialize calls so two simultaneous requests don't overwrite each other's files.
_run_lock = asyncio.Lock()


def extract_text(result: dict) -> str:
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def gcloud_access_token() -> str:
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True,
    ).strip()


@mcp.tool()
async def analyze_bigquery(question: str) -> str:
    """Ask GCP Maestro to analyze BigQuery data in project ia-2006.

    Maestro delegates schema discovery and read-only SQL to its GCP Data
    specialist, and bar/scatter plotting to its Plot Agent. The returned JSON
    contains the answer and any plot file paths. Use this for data questions,
    summaries and charts; do not pass millions of raw records.
    """
    async with _run_lock:
        # New token per invocation: the local OpenCode MCP process can be long-lived.
        transport = StreamableHttpTransport(
            url=MCP_URL,
            headers={
                "Authorization": f"Bearer {gcloud_access_token()}",
                "x-goog-user-project": PROJECT_ID,
            },
        )
        before = {
            p.name: p.stat().st_mtime_ns
            for p in ROOT.glob("*_plot.png")
        }
        async with MCPAdapter(transport) as adapter:
            mcp_tools = await adapter.list_tools()
            data_agent = build_gcp_data_agent(mcp_tools)

            @tool
            async def ask_gcp_data(request: str) -> str:
                """Discover BigQuery datasets/tables or run read-only SQL aggregation."""
                response = await data_agent.ainvoke(
                    {"messages": [{"role": "user", "content": request}]}
                )
                return extract_text(response)

            @tool
            async def ask_plot_agent(request: str) -> str:
                """Create bar or scatter plots using verified numeric data."""
                response = await asyncio.to_thread(
                    plot_agent.invoke,
                    {"messages": [{"role": "user", "content": request}]},
                )
                return extract_text(response)

            maestro = create_agent(
                #model=ChatGoogleGenerativeAI(
                #    model="gemini-3.8-flash", temperature=0,
                #),
                model=ChatOllama(
                    model="llama3.1:latest",
                    temperature=0
                                    ),
                tools=[ask_gcp_data, ask_plot_agent],
                system_prompt="""You are GCP Maestro, orchestrating GCP Data and Plot Agent.
Ask GCP Data to inspect schema and compute small, accurate SQL aggregate
results before plotting. Pass ONLY verified numbers to Plot Agent; NEVER
invent data or claims. SQL must remain read-only. If a tool fails, say so.
For visualization, invoke Plot Agent and report the created plot filename.
Return a concise analysis with the SQL used where available.""",
            )
            result = await maestro.ainvoke({
                "messages": [{"role": "user", "content": question}]
            })

        # Plot Agent's existing tools write filenames relative to process cwd.
        plot_files = [
            str(p.resolve()) for p in ROOT.glob("*_plot.png")
            if p.stat().st_mtime_ns != before.get(p.name)
        ]
        return json.dumps({
            "project": PROJECT_ID,
            "answer": extract_text(result),
            "plot_files": sorted(plot_files),
        }, ensure_ascii=False)


if __name__ == "__main__":
    # FastMCP defaults to stdio, which OpenCode expects for a local MCP server.
    mcp.run()
