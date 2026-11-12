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


@mcp.tool()
def search_governance_knowledge(query: str) -> str:
    """
    Search the Pinecone RAG knowledge base for relevant documents.

    Available knowledge includes:
    - BBVA banking policies and account-opening requirements.
    - Required identification and proof-of-address documents.
    - Data governance, privacy, retention, and metadata policies.

    Use this tool FIRST for questions about banking requirements,
    document validation, account opening, or data governance.

    Returns retrieved evidence with source filenames and page numbers.
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
