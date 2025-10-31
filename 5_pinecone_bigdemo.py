import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings  # only needed if you rebuild the wrapper w/ embedding
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

# --- ENV VARS ---
INDEX_NAME = os.environ["INDEX_NAME"]            # same one used during ingest
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "") # "" (default) unless you set one during ingest
LLM_MODEL  = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:latest")

# --- LLM (local) ---
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# --- VectorStore wrapper (existing Pinecone index) ---
# We only need the embedding class here if the vectorstore constructor requires it for similarity search.
# For PineconeVectorStore (LangChain), passing an embedding func is standard.
emb = OllamaEmbeddings(model=os.getenv("OLLAMA_EMB_MODEL", "bge-large"))  # must match dimensions used to build the index
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=emb,
    namespace=NAMESPACE or None,  # None == default namespace
)

# --- Retriever config ---
# MMR gives diverse top-k; fetch_k controls pool before MMR selection.
# Add an optional metadata filter to restrict by "source" (since TextLoader sets metadata["source"] = filename).
RESTRICT_TO_SOURCE = os.getenv("RESTRICT_TO_SOURCE", "mediumblog1.txt")
metadata_filter = {"source": RESTRICT_TO_SOURCE} if RESTRICT_TO_SOURCE else None

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 24,
        **({"filter": metadata_filter} if metadata_filter else {}),
    },
)

# --- Prompt: ask model to use retrieved context, be concise, and cite sources ---
system_prompt = """You are a concise teaching assistant.
Use the context to answer the user's question accurately.
Cite your sources at the end as [source] where source is the filename or metadata source.
If the answer is not in the context, say you don't know and briefly suggest a follow-up query.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Question: {input}\n\nUse these retrieved notes:\n{context}\n\nAnswer:"),
])

# --- Chain: (Retriever) -> Stuff Docs -> LLM ---
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- Pretty print helpers for demo output ---
def unique_sources(docs):
    srcs = []
    for d in docs:
        s = (d.metadata or {}).get("source")
        if s and s not in srcs:
            srcs.append(s)
    return srcs

def run_query(q: str):
    print(f"\n[Query] {q}")
    res = rag_chain.invoke({"input": q})
    answer = res.get("answer", "").strip()

    # The retrieval_chain returns "context" in res["context"] (list of Documents) in recent LangChain.
    ctx_docs = res.get("context") or res.get("context_documents") or []
    sources = unique_sources(ctx_docs)

    print("\n[Answer]\n", answer)
    if sources:
        print("\n[Sources]")
        for s in sources:
            print(" -", s)

if __name__ == "__main__":
    # --- Demo prompts for class ---
    demo_questions = [
        "In one paragraph, what is a vector database and why is it useful for RAG?",
        "Explain embeddings in plain terms. Include how similarity search works.",
        "What chunk size and overlap are reasonable defaults for PDFs?",
        "What is MMR search and why might it help retrieval quality?",
    ]
    for q in demo_questions:
        run_query(q)

    # Uncomment to allow ad-hoc input:
    # while True:
    #     q = input("\nAsk me something (or 'quit'): ").strip()
    #     if q.lower() in {"quit", "exit"}:
    #         break
    #     run_query(q)
