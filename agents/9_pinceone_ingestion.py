"""
9_ingest_pinecone.py

Build a persistent student-data-governance knowledge base in Pinecone.

Corpus:
- PDF public documents
- TXT internal teaching policies
- CSV data catalog

Pipeline:
files -> LangChain Documents -> chunks -> embeddings -> Pinecone
"""

import csv
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

DOCS_DIR = Path(__file__).parent / "rag_documents"
INDEX_NAME = os.environ["INDEX_NAME"]


def load_pdf(path: Path) -> list[Document]:
    """Load one LangChain Document per PDF page."""
    reader = PdfReader(path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "document_type": "pdf",
                        "page": page_number,
                    },
                )
            )
    return documents


def load_text(path: Path) -> list[Document]:
    """Load a plain-text policy document."""
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={
                "source": path.name,
                "document_type": "text",
            },
        )
    ]


def load_csv(path: Path) -> list[Document]:
    """Turn each catalog row into a readable Document."""
    documents = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_number, row in enumerate(reader, start=2):
            content = "\n".join(f"{key}: {value}" for key, value in row.items())

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "document_type": "data_catalog",
                        "row": row_number,
                        "field_name": row.get("field_name", ""),
                    },
                )
            )

    return documents


def load_corpus(folder: Path) -> list[Document]:
    """Load all supported files from the governance corpus."""
    documents = []

    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            documents.extend(load_pdf(path))
        elif suffix == ".txt":
            documents.extend(load_text(path))
        elif suffix == ".csv":
            documents.extend(load_csv(path))

    return documents


def stable_chunk_id(doc: Document) -> str:
    """Create a deterministic ID so rerunning ingestion does not duplicate chunks."""
    identity = (
        f"{doc.metadata.get('source', '')}|"
        f"{doc.metadata.get('page', '')}|"
        f"{doc.metadata.get('row', '')}|"
        f"{doc.page_content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(f"Loading governance documents from: {DOCS_DIR}")

    documents = load_corpus(DOCS_DIR)
    print(f"Loaded {len(documents)} source documents/pages/rows")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model="bge-large")

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    ids = [stable_chunk_id(chunk) for chunk in chunks]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(f"Ingested {len(chunks)} chunks into Pinecone index: {INDEX_NAME}")
