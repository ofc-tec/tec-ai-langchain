"""
15_drive_mcp_client.py

Google-managed Drive MCP client.

Demonstrates:
    1. Authentication using Google ADC
    2. MCP tool discovery (tools/list)
    3. MCP tool execution (tools/call)
    4. MCP error diagnostics

Run:
    uv run python mcp/15_drive_mcp_client.py
"""

import asyncio

import google.auth
from google.auth.transport.requests import Request

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.exceptions import MCPError


PROJECT_ID = "ia-2006"
DRIVE_MCP_URL = "https://drivemcp.googleapis.com/mcp/v1"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

# Previously verified Drive folder
FILE_ID = "1qFZbwDebU2YTSWyIptz08714xWpbM645uyHvDE2YZnU"


from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

TOKEN_FILE = Path.home() / ".config/opencode/drive-mcp-token.json"

def get_drive_token() -> str:
    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        scopes=[DRIVE_SCOPE]
    )

    if not credentials.valid:
        credentials.refresh(Request())

    return credentials.token

async def main():

    # 1. Connect to Google's managed MCP server
    transport = StreamableHttpTransport(
        url=DRIVE_MCP_URL,
        headers={
            "Authorization": f"Bearer {get_drive_token()}",
            "x-goog-user-project": PROJECT_ID,
        },
    )

    async with Client(transport) as client:

        # 2. Discover tools
        tools = await client.list_tools()

        print(f"\nConnected to Google Drive MCP — {len(tools)} tools:\n")

        for tool in tools:
            print(f"  - {tool.name}")

        # 3. Test get_file_metadata
        print("\n" + "=" * 60)
        print("TEST: tools/call → get_file_metadata")
        print("=" * 60)

        try:
            result = await client.call_tool(
                "get_file_metadata",
                {
                    "fileId": FILE_ID,
                    "excludeContentSnippets": True,
                },
            )

            print("\nTool execution successful!\n")
            print(result)

        # 4. Error diagnostics
        except MCPError as e:
            print("\nMCP TOOL ERROR")
            print("-" * 40)
            print("Error code:", e.code)
            print("Error message:", e.message)
            print("Error data:", e.data)

        except Exception as e:
            print("\nUNEXPECTED ERROR")
            print("-" * 40)
            print("Error type:", type(e).__name__)
            print("Error message:", str(e))


if __name__ == "__main__":
    asyncio.run(main())