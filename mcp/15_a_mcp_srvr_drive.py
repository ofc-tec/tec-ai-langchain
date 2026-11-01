"""
15_a_mcp_srvr_drive.py

Local MCP proxy for Google's managed Drive MCP server.

OpenCode <--stdio--> FastMCP Proxy <--HTTPS--> Google Drive MCP

uv run fastmcp dev mcp/15_a_mcp_srvr_drive.py 
"""

import google.auth
from google.auth.transport.requests import Request

from fastmcp.server import create_proxy
from fastmcp.client.transports import StreamableHttpTransport


PROJECT_ID = "ia-2006"
DRIVE_MCP_URL = "https://drivemcp.googleapis.com/mcp/v1"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
TOKEN_FILE = Path.home() / ".config/opencode/drive-mcp-token.json"

def get_drive_token() -> str:
    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        scopes=[DRIVE_SCOPE]
    )

    if not credentials.valid:
        credentials.refresh(Request())

    return credentials.token

# Connect to Google's existing MCP server.
transport = StreamableHttpTransport(
    url=DRIVE_MCP_URL,
    headers={
        "Authorization": f"Bearer {get_drive_token()}",
        "x-goog-user-project": PROJECT_ID,
    },
)

# Expose Google's MCP tools through our local server.
mcp = create_proxy(
    transport,
    name="Google Drive MCP Proxy",
)


if __name__ == "__main__":
    mcp.run(transport="stdio")