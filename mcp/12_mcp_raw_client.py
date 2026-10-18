"""
12_mcp_raw_client.py

Minimal MCP 2.x client to verify the server before adding LangChain.
"""

import asyncio
from pathlib import Path

from mcp import Client, StdioServerParameters

SERVER_FILE = Path(__file__).parent / "11_mcp_rag_server.py"


async def main() -> None:
    server = StdioServerParameters(
        command="uv",
        args=["run", "python", str(SERVER_FILE)],
    )

    async with Client(server) as client:
        tools_result = await client.list_tools()

        print("\n--- MCP TOOLS ---")
        for tool in tools_result.tools:
            print(f"{tool.name}: {tool.description}")

        question = "Can face embeddings be used for student marketing?"

        result = await client.call_tool(
            "search_governance_knowledge",
            {"query": question},
        )

        print("\n--- TOOL RESULT ---")
        for block in result.content:
            if getattr(block, "type", None) == "text":
                print(block.text)


if __name__ == "__main__":
    asyncio.run(main())
