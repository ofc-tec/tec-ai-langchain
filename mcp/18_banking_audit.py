"""
18_banking_audit.py

Read-only MCP server for auditing BigQuery activity.

Run:
    uv run python mcp/18_banking_audit.py
"""

import json
from datetime import datetime, timedelta, timezone

from fastmcp import FastMCP
from google.cloud import logging_v2

PROJECT_ID = "ia-2006"

mcp = FastMCP("Banking Audit")


@mcp.tool()
def audit_bigquery(days: int = 1, limit: int = 10) -> str:
    """Retrieve recent BigQuery audit events.
    if none is provided  assume 1 day and 10 records.
    avoid multiple calls, answer the question in as few calls as possible, if the user asks for more than 7 days, limit to 7 days. (tell him that you limited to 7 days).could
    Args:
        days: Number of days to inspect (1-7).
        

    Returns:
        Recent BigQuery audit records.
    """

    days = max(1, min(days, 7))
    limit = max(1, min(limit, 50))

    client = logging_v2.Client(project=PROJECT_ID)

    start = datetime.now(timezone.utc) - timedelta(days=days)

    log_filter = (
        'protoPayload.serviceName="bigquery.googleapis.com" '
        f'AND timestamp>="{start.isoformat()}"'
    )

    entries = client.list_entries(
        filter_=log_filter,
        order_by="timestamp desc",
        max_results=limit,
    )

    results = []

    for entry in entries:
        results.append({
            "timestamp": str(entry.timestamp),
            "log_name": entry.log_name,
            "payload": entry.payload,
        })

    return json.dumps(results, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()