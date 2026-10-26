
import vertexai
from vertexai import rag

PROJECT = "ia-2006"
REGION = "us-central1"

CORPUS = (
    "projects/ia-2006/locations/us-central1/"
    "ragCorpora/5553294582315220992"
)

vertexai.init(project=PROJECT, location=REGION)

response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(rag_corpus=CORPUS)
    ],
    text="Can I use students data for advertising?",
    rag_retrieval_config=rag.RagRetrievalConfig(top_k=3),
)

print(response)
