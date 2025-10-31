#!/usr/bin/env python3
# filename: agent_with_faiss_and_tavily.py

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.documents import Document

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS

# >>>>>>>>>>>>>>>>>>>> CONFIG <<<<<<<<<<<<<<<<<<<<
EMBED_MODEL = "nomic-embed-text"   # <-- change to the one you pulled (e.g., "bge-m3", "mxbai-embed-large")
CHAT_MODEL  = "llama3.1:latest"    # tool-capable
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@tool
def get_text_length(text: str) -> int:
    """Return length of a text by characters."""
    return len(text)

def build_demo_faiss(emb):
    docs = [
        Document(page_content="hello world", metadata={"source": "demo:1"}),
        Document(page_content="langchain is neat", metadata={"source": "demo:2"}),
        Document(page_content="faiss local index", metadata={"source": "demo:3"}),
        Document(page_content="visual SLAM with HMMs is discussed in our lab notes", metadata={"source": "demo:4"}),
    ]
    return FAISS.from_documents(docs, emb)

# Global vector store for the tool
VS: FAISS | None = None

@tool
def search_local_docs(query: str, k: int = 4) -> str:
    """Search the local FAISS vector DB and return top-k snippets."""
    if VS is None:
        return "Local vector DB is not initialized."
    if hasattr(VS, "similarity_search_with_score"):
        hits = VS.similarity_search_with_score(query, k=k)
        return "\n---\n".join(
            f"[score={score:.3f}] {(d.metadata or {}).get('source','')}\n{d.page_content}"
            for d, score in hits
        ) or "No local matches."
    docs = VS.similarity_search(query, k=k)
    return "\n---\n".join(
        f"{(d.metadata or {}).get('source','')}\n{d.page_content}" for d in docs
    ) or "No local matches."

def main():
    # LLM & tools
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    tavily = TavilySearchResults(max_results=5)  # needs TAVILY_API_KEY in .env
    tools = [get_text_length, search_local_docs, tavily]

    # Prompt (must include agent_scratchpad)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when beneficial."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Embeddings + FAISS (Ollama only)
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    global VS
    VS = build_demo_faiss(emb)

    # Demo queries
    print("\n=== Local Retrieval ===")
    print(agent_executor.invoke({"input": "Use the local docs to list the strings in our FAISS index."}))

    print("\n=== Web + Local ===")
    print(agent_executor.invoke({"input": "Find two recent venues for HMM-based visual SLAM; check local notes too."}))

if __name__ == "__main__":
    main()
