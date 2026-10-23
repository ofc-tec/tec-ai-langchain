"""Expose GCP Maestro as an MCP tool over stdio.

Run from the repository root:
    uv run python agents/maestro_mcp_server.py

Model choices (Gemini/Ollama), tools, and orchestration live in gcp_maestro.py.
Authentication and the upstream BigQuery MCP connection live in gcp_data.py.
"""
from fastmcp import FastMCP

from gcp_maestro import build_gcp_maestro, extract_text

mcp = FastMCP("GCP Maestro")


@mcp.tool()
def analyze_bigquery(question: str) -> str:
    """Ask GCP Maestro for BigQuery analysis and optional bar/scatter charts."""
    maestro = build_gcp_maestro()
    result = maestro.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": "maestro-mcp-demo"}},
    )
    return extract_text(result)


if __name__ == "__main__":
    mcp.run()
