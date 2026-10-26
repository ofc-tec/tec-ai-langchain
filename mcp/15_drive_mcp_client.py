"""Discover tools on Google's managed Drive MCP server using existing ADC.

Prerequisites:
    gcloud auth application-default login --client-id-file=... \
        --scopes=https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/cloud-platform
    uv add fastmcp google-auth

Run:
    uv run python 15_drive_mcp_client.py
"""

import asyncio

import google.auth
from google.auth.transport.requests import Request
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

PROJECT_ID = "ia-2006"
DRIVE_MCP_URL = "https://drivemcp.googleapis.com/mcp/v1"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def get_drive_token() -> str:
    """Reuse locally saved Application Default Credentials; refresh when needed."""
    credentials, _ = google.auth.default(scopes=[DRIVE_SCOPE])
    if not credentials.valid:
        credentials.refresh(Request())
    if not credentials.token:
        raise RuntimeError("Google ADC did not provide an access token")
    return credentials.token


async def main() -> None:
    transport = StreamableHttpTransport(
        url=DRIVE_MCP_URL,
        headers={
            "Authorization": f"Bearer {get_drive_token()}",
            "x-goog-user-project": PROJECT_ID,
        },
    )

    async with Client(transport) as client:
        tools = await client.list_tools()
        print(f"Connected to Google Drive MCP — {len(tools)} tools:\n")
        for tool in tools:
            print(f"  - {tool.name}")


if __name__ == "__main__":
    asyncio.run(main())
