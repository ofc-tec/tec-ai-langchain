from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

# Existing RAG Engine corpus
RAG_CORPUS = (
    "projects/ia-2006/locations/us-central1/"
    "ragCorpora/5553294582315220992"
)

# Google Cloud RAG Engine retrieval tool
rag_tool = VertexAiRagRetrieval(
    name="retrieve_policies",
    description=(
        "Retrieves institutional policies about data governance, "
        "student privacy, data retention, and responsible AI usage."
    ),
    rag_resources=[
        rag.RagResource(rag_corpus=RAG_CORPUS)
    ],
    similarity_top_k=5,
)

# ADK agent
root_agent = Agent(
    name="gcp_rag_engine_agent",
    model="gemini-2.5-flash",
    description="Institutional policy assistant grounded in Google Cloud RAG Engine.",
    instruction="""
You are an institutional data governance assistant.

Your task is to answer questions about institutional policies
using documents stored in Google Cloud RAG Engine.

INSTRUCTIONS:
1. Always call retrieve_policies before answering policy questions.
2. Base your answers on retrieved document passages.
3. Cite source document names when available.
4. Never invent institutional policies or requirements.
5. If the retrieved documents do not contain the answer,
   explicitly acknowledge the missing information.
6. Distinguish documented policy requirements from recommendations.

Answer clearly and concisely.
""",
    tools=[rag_tool],
)