from fastmcp.server import create_proxy

# Reuse the authenticated transport configuration
# from 15_drive_mcp_client.py.
# We need to extract that setup into a reusable function.

from drive_transport import make_drive_transport

proxy = create_proxy(
    make_drive_transport(),
    name="Google Drive MCP Proxy"
)

if __name__ == "__main__":
    proxy.run(transport="stdio")