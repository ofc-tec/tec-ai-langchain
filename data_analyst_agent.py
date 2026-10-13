"""
data_analyst_agent.py

LangChain data-analyst agent for BigQuery.

Agent tools:
    ONLY tools discovered from Google's official BigQuery remote MCP server.

Application-side rendering:
    matplotlib is used only to render chart data already returned by the agent.
    It is NOT exposed to the agent as a tool.

Prerequisites
-------------
1. Enable the BigQuery API in your Google Cloud project.
2. Authenticate Application Default Credentials:

       gcloud auth application-default login

3. Set your billing/quota project:

       
       export GOOGLE_CLOUD_PROJECT="your-project-id"
       e.g.
       export GOOGLE_CLOUD_PROJECT="ia-2006"

4. Install:

       pip install "langchain[mcp]>=1.4.0" \
                   langchain-openai \
                   google-auth \
                   matplotlib

5. Set the model API key, for example:

       export OPENAI_API_KEY="..."

Optional:
       export DATA_ANALYST_MODEL="openai:gpt-5-nano"

Run:
       python data_analyst_agent.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import google.auth
from google.auth.transport.requests import Request

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.memory import InMemorySaver


BIGQUERY_MCP_URL = "https://bigquery.googleapis.com/mcp"
BIGQUERY_SCOPE = "https://www.googleapis.com/auth/bigquery"

MODEL = os.getenv("DATA_ANALYST_MODEL", "openai:gpt-5-nano")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


SYSTEM_PROMPT = """
You are a careful BigQuery Data Analyst.

You have access ONLY to tools supplied by Google's official BigQuery MCP server.
Use those tools as your source of truth for datasets, tables, schemas, metadata,
and query results.

Your workflow for an analysis request is:

1. Discover relevant datasets when the user has not named one.
2. Inspect available tables.
3. Inspect table metadata/schema before writing SQL.
4. Form a small analytical question.
5. Use aggregate SQL whenever possible.
6. Avoid SELECT * over large tables.
7. Perform a compact EDA:
   - understand table scale when useful,
   - inspect important categorical distributions,
   - inspect numeric ranges / averages when useful,
   - inspect missingness when useful,
   - identify one or two interesting relationships.
8. Explain what the data supports and what it does NOT support.
9. Never invent a dataset, table, column, statistic, or query result.

Query safety:
- Prefer read-only analytical SQL.
- Keep exploratory result sets small.
- Use LIMIT when returning row-level examples.
- Do not modify or delete BigQuery resources.

Visualization:
If a plot would materially help, first obtain the required aggregated data using
the BigQuery MCP tools. Then include ONE machine-readable block at the END of
your final answer in exactly this form:

```chart
{
  "chart_type": "bar",
  "title": "A short title",
  "x": "column_name",
  "y": "column_name",
  "rows": [
    {"column_name": "value", "other_column": 123}
  ]
}
```

Allowed chart_type values are: bar, line, scatter.

Only include a chart block when the returned data is small enough to plot
clearly (normally <= 50 rows). The rows must come from actual BigQuery results.
Do not fabricate chart data.

Outside that chart block, give a concise analyst-style explanation.
"""


def get_google_credentials() -> tuple[str, str]:
    """
    Load ADC, refresh an OAuth access token, and resolve the quota project.

    Google remote MCP servers require:
      Authorization: Bearer <token>
      X-Goog-User-Project: <project>
    """
    credentials, adc_project = google.auth.default(scopes=[BIGQUERY_SCOPE])
    credentials.refresh(Request())

    project_id = PROJECT_ID or adc_project
    if not project_id:
        raise RuntimeError(
            "No Google Cloud project found. Set GOOGLE_CLOUD_PROJECT."
        )

    if not credentials.token:
        raise RuntimeError("Could not obtain a Google OAuth access token.")

    return credentials.token, project_id


def build_bigquery_mcp_client() -> Client:
    """Create an authenticated FastMCP client for Google's BigQuery MCP server."""
    token, project_id = get_google_credentials()

    transport = StreamableHttpTransport(
        url=BIGQUERY_MCP_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Goog-User-Project": project_id,
        },
    )
    return Client(transport)


def extract_chart_spec(text: str) -> dict[str, Any] | None:
    """
    Extract the optional ```chart ... ``` JSON block from the agent response.
    """
    match = re.search(r"```chart\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def render_chart(spec: dict[str, Any]) -> str | None:
    """
    Render chart data already present in the final response.

    IMPORTANT:
    This is application-side presentation code, NOT an agent tool.
    All factual data must already have come through the BigQuery MCP server.
    """
    import matplotlib.pyplot as plt

    chart_type = spec.get("chart_type")
    title = spec.get("title", "BigQuery analysis")
    x = spec.get("x")
    y = spec.get("y")
    rows = spec.get("rows", [])

    if chart_type not in {"bar", "line", "scatter"}:
        return None
    if not x or not y or not rows:
        return None
    if any(x not in row or y not in row for row in rows):
        return None

    xs = [row[x] for row in rows]
    ys = [row[y] for row in rows]

    plt.figure(figsize=(9, 5))

    if chart_type == "bar":
        plt.bar(xs, ys)
        plt.xticks(rotation=45, ha="right")
    elif chart_type == "line":
        plt.plot(xs, ys, marker="o")
        plt.xticks(rotation=45, ha="right")
    else:
        plt.scatter(xs, ys)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()

    output_path = "data_analyst_plot.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


async def main() -> None:
    mcp_client = build_bigquery_mcp_client()

    # MCPAdapter converts Google's MCP tools into ordinary LangChain tools.
    async with MCPAdapter(mcp_client) as adapter:
        tools = await adapter.list_tools()

        print("\nBIGQUERY MCP TOOLS")
        print("------------------")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        # This is the modern LangChain agent API.
        agent = create_agent(
            model=MODEL,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

    # One persistent classroom conversation.
    config = {
        "configurable": {
            "thread_id": "bigquery-data-analyst-classroom"
        }
    }

    print("\nBIGQUERY DATA ANALYST")
    print("---------------------")
    print(f"Model: {MODEL}")
    print("Examples:")
    print("  What datasets can I access?")
    print("  Explore one interesting public dataset and give me a compact EDA.")
    print("  Find a useful table, analyze one categorical distribution, and plot it.")
    print("  Inspect the available e-commerce data and suggest three analytical questions.")
    print("\nType 'quit' to exit.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        if not user_text:
            continue

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_text,
                    }
                ]
            },
            config=config,
        )

        final_message = result["messages"][-1]
        text = getattr(final_message, "text", None) or str(final_message.content)

        print("\nAgent:")
        print(text)

        chart_spec = extract_chart_spec(text)
        if chart_spec:
            path = render_chart(chart_spec)
            if path:
                print(f"\n[application renderer] chart saved to: {path}")
            else:
                print("\n[application renderer] chart block was invalid.")
        print()


if __name__ == "__main__":
    asyncio.run(main())
