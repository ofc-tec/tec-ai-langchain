#!/usr/bin/env python3
# filename: 3_FAISS_RAG.py

from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# -----------------------------
# LLM + Embeddings (Ollama-only)
# -----------------------------
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:latest")
EMB_MODEL = os.getenv("OLLAMA_EMB_MODEL", "nomic-embed-text")  ## OLLAMA PULL to get it-

llm = ChatOllama(model=LLM_MODEL, temperature=0)
emb = OllamaEmbeddings(model=EMB_MODEL)

# -----------------------------
# Build / load FAISS from PDF
# -----------------------------
PDF_PATH = "react.pdf"
INDEX_DIR = "faiss_index_react"

def ensure_vectorstore() -> FAISS:
    if os.path.isdir(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

    print("[ingest] loading PDF…")
    docs = PyPDFLoader(PDF_PATH).load()

    print("[ingest] splitting…")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    chunks = splitter.split_documents(docs)

    print(f"[ingest] {len(chunks)} chunks  FAISS")
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(INDEX_DIR)
    return vs

vectorstore = ensure_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# Tools
# -----------------------------
@tool
def get_text_length(text: str) -> int:
    """Return length of a text by characters."""
    return len(text)

@tool
def retrieve(text: str) -> str:
    """
    Retrieve the most relevant passages from the indexed PDF.
    Return a compact text block with brief citations.
    """
    docs = retriever.invoke(text)### ALL THE VECTORDB MAGIC HAPPENS HERE
    out_lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        page = meta.get("page", "?")
        out_lines.append(f"[{i}] (page {page}) {d.page_content.strip()}")
    return "\n\n".join(out_lines)

tools = [get_text_length, retrieve]

# -----------------------------
# Agent (tool-calling style)
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a concise, helpful assistant. "
     "When a question involves the PDF, ALWAYS call the `retrieve` tool first "
     "to gather context, then answer using that context. "
     "Cite snippets by their [n] tag from the retrieve output when relevant."
    ),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    # Simple tool demo (unchanged from your working file)
    res1 = agent_executor.invoke({"input": "Call get_text_length on 'hola'."})
    print("\n[get_text_length demo]\n", res1)

    q = "Give me the gist of ReAct in 3 sentences"
    res2 = agent_executor.invoke({"input": q})
    print("\n[RAG answer]\n", res2["output"])

