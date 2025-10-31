import os
from dotenv import load_dotenv

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

if __name__ == "__main__":
    print("Retrieving...")

    index_name = os.getenv("PINECONE_INDEX") or os.getenv("INDEX_NAME")
    if not index_name:
        raise RuntimeError("Set PINECONE_INDEX or INDEX_NAME in .env")

    embeddings = OllamaEmbeddings(model="bge-large")  # 1024 dims to match your index
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    llm = ChatOllama(temperature=0, model="llama3:instruct")    
    
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    query = "what is Pinecone in machine learning?"
    result = chain.invoke({"input": query})

    print("\nANSWER:\n", result.get("answer"))
    print("\nSOURCES:")
    for d in result.get("context", []):
        print("-", d.metadata.get("source", "(no source)"))
