"""
GCP Data Agent MCP Server.

Exposes the existing LangChain GCP Data Agent to OpenCode.
"""

from fastmcp import FastMCP
from gcp_data import run_gcp_data

mcp = FastMCP("GCP Data Agent")


@mcp.tool()
async def analyze_bigquery(question: str) -> str:
    """
    Ask the GCP Data Agent to analyze BigQuery.

    Supports dataset discovery, schema inspection,
    SQL queries, and numerical analysis.
    """
    return await run_gcp_data(question)


if __name__ == "__main__":
    mcp.run()