"""13 - LangChain Data Steward using the MCP RAG server (MCP 2.x).

Requires: langchain[mcp], langchain-google-genai, working file 11,
Pinecone credentials, local Ollama embeddings, and GEMINI_API_KEY.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_google_genai import ChatGoogleGenerativeAI

HERE = Path(__file__).resolve().parent
SERVER_FILE = HERE / "11_mcp_rag_server.py"

SYSTEM_PROMPT = """You are a Data Steward assistant for a teaching exercise.
When asked about student-data classification, ownership, privacy, retention,
allowed uses, or AI policy, use the governance search tool before answering.
Ground governance conclusions in retrieved evidence and cite source filenames.
Clearly identify synthetic teaching examples as nonofficial policies.
If evidence is missing or contradictory, say so; do not invent permissions.
Respond to the user concisely rather than dumping retrieved chunks.
"""


def content_as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


async def main():
    load_dotenv(HERE.parent / ".env")
    load_dotenv(HERE / ".env")

    if not SERVER_FILE.is_file():
        raise FileNotFoundError(f"MCP server not found: {SERVER_FILE}")

    # MCPAdapter takes a server target, NOT a `mcp.Client` instance.
    # The script path makes its FastMCP client launch the local server
    # automatically using stdio. Do not pass env=dict(os.environ):
    # a failed connection can echo credentials into the traceback.
    async with MCPAdapter(SERVER_FILE) as adapter:
        tools = await adapter.list_tools()

        print("\n--- DISCOVERED MCP TOOLS ---")
        for tool in tools:
            print(f"- {tool.name}")

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            temperature=0,
        )
        agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

        question = (
            "Can face embeddings be used for student marketing? "
            "Who owns the field and what is its retention period?"
        )
        print("\n--- QUESTION ---")
        print(question)

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })

        print("\n--- AGENT TOOL CALLS ---")
        calls = [
            call for message in result["messages"]
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        for call in calls:
            print(f"{call['name']}({call['args']})")
        if not calls:
            print("(None)")

        print("\n--- FINAL ANSWER ---")
        print(content_as_text(result["messages"][-1].content))


if __name__ == "__main__":
    asyncio.run(main())
