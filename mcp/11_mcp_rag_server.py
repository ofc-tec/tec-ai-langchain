"""
11_mcp_rag_server.py

MCP 2.x server that exposes the existing Pinecone RAG retriever as an MCP tool.
"""

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

#from mcp.server import MCPServer
#mcp = MCPServer("Student Governance RAG")  # Deprecation

from fastmcp import FastMCP
mcp = FastMCP("Student Governance RAG")

embeddings = OllamaEmbeddings(model="bge-large")

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
)


@mcp.tool()  #<-- This decorator registers the function as a tool in the MCP server (not the agent)
def search_governance_knowledge(query: str) -> str:
    """
    Search student-data governance documents for relevant evidence.

    Use this tool for questions about privacy, data ownership, retention,
    approved use, AI use, or other governance rules.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant governance evidence was found."

    evidence = []
#############FOr debuggin purposes
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        row = doc.metadata.get("row")
        location = source
        if page is not None:
            location += f", page {page}"
        if row is not None:
            location += f", row {row}"
        evidence.append(
            f"SOURCE: {location}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(evidence)

if __name__ == "__main__":
    mcp.run()       #  <----- run the MCP server on localhost:8000 (default)
