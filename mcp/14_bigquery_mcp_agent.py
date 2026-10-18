"""
14_bigquery_mcp_agent.py
Google-managed BigQuery MCP -> LangChain agent (read-only teaching demo).

Run:
    uv run python mcp/14_bigquery_mcp_agent.py

Prerequisites:
    gcloud auth login
    gcloud config set project ia-2006
    GOOGLE_API_KEY / GEMINI_API_KEY available to LangChain

The access token is short-lived. Re-run this script to refresh it.
"""
import asyncio
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp.client.transports import StreamableHttpTransport
from googleapiclient import model
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

PROJECT_ID = "ia-2006"
BIGQUERY_MCP_URL = "https://bigquery.googleapis.com/mcp"
# These are the ONLY server tools the model can call.
ALLOWED_TOOLS = {
    "list_dataset_ids",
    "get_dataset_info",
    "list_table_ids",
    "get_table_info",
}
# Enable SQL only after metadata discovery is working.
ENABLE_READONLY_SQL = False
if ENABLE_READONLY_SQL:
    ALLOWED_TOOLS.add("execute_sql_readonly")

SYSTEM_PROMPT = f"""You are a careful BigQuery metadata discovery assistant.
Only explore project {PROJECT_ID}, unless explicitly asked otherwise.
Begin by listing available datasets with the provided tools.
To describe tables, inspect their metadata and schema using MCP tools.
Never invent dataset names, table names, fields, or query results.
Do not attempt to modify any data.
Only cite details returned by the tools. Summarize concisely.
If a permission error occurs, report which operation failed.
"""


def gcloud_access_token() -> str:
    # Do not log the token; only capture its stdout.
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\\n".join(
            x.get("text", str(x)) if isinstance(x, dict) else str(x)
            for x in content
        )
    return str(content)


async def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = gcloud_access_token()

    transport = StreamableHttpTransport(
        url=BIGQUERY_MCP_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
        },
    )

    async with MCPAdapter(transport) as adapter:
        all_tools = await adapter.list_tools()
        print("\\n--- AVAILABLE MCP TOOLS ---")
        print(", ".join(t.name for t in all_tools))

        tools = [tool for tool in all_tools if tool.name in ALLOWED_TOOLS]
        names = {tool.name for tool in tools}
        if "list_dataset_ids" not in names:
            raise RuntimeError("BigQuery MCP did not provide list_dataset_ids")

        print("\\n--- TOOLS GIVEN TO AGENT ---")
        print(", ".join(sorted(names)))

        #model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        #llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        llm = ChatOllama(
                model="llama3.1:latest",
                temperature=0
                                )

        agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

        question = (
            f"List the datasets that I can access in Google Cloud project "
            f"{PROJECT_ID}. For up to 5 datasets, briefly describe what "
            "they contain using available dataset metadata. "
            "Do not execute SQL."
        )
        print("\\n--- QUESTION ---\\n" + question)

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })

        print("\\n--- AGENT TOOL CALLS ---")
        for message in result["messages"]:
            for call in getattr(message, "tool_calls", []) or []:
                print(f"{call['name']}({call['args']})")

        print("\\n--- FINAL ANSWER ---")
        print(as_text(result["messages"][-1].content))


if __name__ == "__main__":
    asyncio.run(main())
