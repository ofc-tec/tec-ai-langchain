"""
9_datasteward_rag.py

Evolution from 7_first_data_steward.py:
- Keep deterministic tools for structured governance facts.
- Replace hard-coded policy lookup with RAG over governance documents.
- The agent decides when it needs the RAG tool.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings


DATA_CATALOG = {
    "customer_income": {
        "description": "Estimated annual income of the customer.",
        "owner": "Retail Risk Data Office",
        "classification": "Sensitive",
    },
    "customer_id": {
        "description": "Internal unique identifier for a customer.",
        "owner": "Customer Platform",
        "classification": "Confidential",
    },
    "email": {
        "description": "Customer email address used for communications.",
        "owner": "Customer Experience",
        "classification": "Personal",
    },
    "credit_score": {
        "description": "Internal or external credit-risk score.",
        "owner": "Credit Risk",
        "classification": "Sensitive",
    },
}


governance_documents = [
    Document(
        page_content=(
            "Marketing Policy: Sensitive financial attributes must not be used "
            "for marketing campaigns unless explicit governance approval has been granted."
        ),
        metadata={"source": "marketing_policy"},
    ),
    Document(
        page_content=(
            "Access Policy: Access to Confidential or Sensitive data requires "
            "approval from the responsible data owner."
        ),
        metadata={"source": "access_policy"},
    ),
    Document(
        page_content=(
            "Privacy Policy: Personal data must only be used for an approved "
            "business purpose and according to the organization's privacy policy."
        ),
        metadata={"source": "privacy_policy"},
    ),
    Document(
        page_content=(
            "Data Quality Policy: A data issue should be opened when a governed "
            "field has missing ownership, conflicting definitions, or repeated "
            "quality failures that cannot be resolved by the consuming team."
        ),
        metadata={"source": "data_quality_policy"},
    ),
]


# Build local RAG knowledge base
embeddings = OllamaEmbeddings(model="bge-large")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=governance_documents)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


@tool
def lookup_glossary(field_name: str) -> str:
    """Look up the description and classification of a data field."""
    key = field_name.strip().lower()

    if key not in DATA_CATALOG:
        return f"No glossary entry was found for '{field_name}'."

    item = DATA_CATALOG[key]

    return (
        f"Field: {key}\n"
        f"Description: {item['description']}\n"
        f"Classification: {item['classification']}"
    )


@tool
def get_data_owner(field_name: str) -> str:
    """Return the data owner responsible for a field."""
    key = field_name.strip().lower()

    if key not in DATA_CATALOG:
        return f"No owner was found for '{field_name}'."

    return f"The data owner for '{key}' is {DATA_CATALOG[key]['owner']}."


@tool
def search_governance_knowledge(question: str) -> str:
    """Search customer governance documents for relevant policies and rules."""
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant governance information was found."

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


@tool
def create_issue(description: str) -> str:
    """Simulate creating a governance issue."""
    return (
        "SIMULATED GOVERNANCE ISSUE CREATED\n"
        f"Description: {description}\n"
        "Status: OPEN"
    )


tools = [
    lookup_glossary,
    get_data_owner,
    search_governance_knowledge,
    create_issue,
]


llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Data Steward agent.

Your job is to answer questions about:
- data meaning,
- data ownership,
- data classification,
- governance policies,
- possible governance issues.

Rules:
1. Use tools instead of inventing governance facts.
2. Use lookup_glossary for field meaning and classification.
3. Use get_data_owner for ownership information.
4. Use search_governance_knowledge when a policy or governance rule is needed.
5. If you need more than one fact, call more than one tool.
6. If the available governance information is insufficient, say so.
7. Do not claim that an issue was created unless you actually call create_issue.
8. Keep the final answer concise and explain the governance recommendation.
""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
)


question = (
    "Can we use customer_income for a marketing campaign? "
    "Who owns this field and who should approve access?"
)

result = executor.invoke({"input": question})

print("\nFINAL ANSWER")
print("------------")
print(result["output"])


# Suggested student tests:
# 1. "What does credit_score mean and who owns it?"
# 2. "Can email be used for any business purpose?"
# 3. "Who should approve access to customer_income?"
# 4. "What policy applies if I want to use customer_income for marketing?"
# 5. Ask about a policy that does NOT exist in the governance corpus.
#
# STUDENT CHALLENGE:
# Add one new governance document.
# Ask a question that requires both a deterministic tool and the RAG tool.
