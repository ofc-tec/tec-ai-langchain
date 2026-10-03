"""
7_first_data_steward.py

Week 7 project draft:
A first LangChain Data Steward agent with deterministic tools.

Important:
- NO RAG yet.
- The agent can only use the small synthetic governance knowledge below.
- Next week, some of these tools can be replaced by retrieval over real/synthetic documents.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent


# ---------------------------------------------------------------------
# 1. Synthetic governance data
# ---------------------------------------------------------------------

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

POLICIES = {
    "marketing": (
        "Sensitive financial attributes must not be used for marketing "
        "without explicit governance approval."
    ),
    "personal_data": (
        "Personal data must only be used for an approved business purpose "
        "and according to the organization's privacy policy."
    ),
    "access": (
        "Access to Confidential or Sensitive data requires approval from "
        "the responsible data owner."
    ),
}


# ---------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------

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
def check_policy(topic: str) -> str:
    """
    Return the governance policy for a known topic.
    Valid topics include: marketing, personal_data, access.
    """
    key = topic.strip().lower().replace(" ", "_")

    if key not in POLICIES:
        return (
            f"No policy was found for '{topic}'. "
            f"Available topics: {', '.join(POLICIES.keys())}."
        )

    return f"Policy '{key}': {POLICIES[key]}"


@tool
def create_issue(description: str) -> str:
    """
    Simulate creating a governance issue.
    This does not write to an external system yet.
    """
    return (
        "SIMULATED GOVERNANCE ISSUE CREATED\n"
        f"Description: {description}\n"
        "Status: OPEN"
    )


tools = [
    lookup_glossary,
    get_data_owner,
    check_policy,
    create_issue,
]


# ---------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ---------------------------------------------------------------------
# 4. Agent prompt
# ---------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a first-version Data Steward agent.

Your job is to answer questions about:
- data meaning,
- data ownership,
- data classification,
- governance policies,
- possible governance issues.

Rules:
1. Use tools instead of inventing governance facts.
2. If you need more than one fact, call more than one tool.
3. Clearly distinguish facts returned by tools from your interpretation.
4. If information is missing, say so.
5. Do not claim that an issue was created unless you actually call create_issue.
6. Keep the final answer concise and explain the governance recommendation.
""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# ---------------------------------------------------------------------
# 5. Agent
# ---------------------------------------------------------------------

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # students can see the agent/tool trajectory
    max_iterations=6,
    handle_parsing_errors=True,
)


# ---------------------------------------------------------------------
# 6. Try it
# ---------------------------------------------------------------------

question = (
    "Can we use customer_income for a marketing campaign? "
    "Who owns this field and who should approve access?"
)

result = executor.invoke({"input": question})

print("\nFINAL ANSWER")
print("------------")
print(result["output"])


# ---------------------------------------------------------------------
# Suggested student tests
# ---------------------------------------------------------------------
#
# 1. "What does credit_score mean and who owns it?"
#
# 2. "Can email be used for any purpose?"
#
# 3. "Who should approve access to customer_income?"
#
# 4. "I want to use customer_income for marketing.
#     Check the relevant governance information."
#
# 5. "Create a governance issue because a Sensitive field
#     is being used without owner approval."
#
#
# STUDENT CHALLENGE
# -----------------
# Add one new field to DATA_CATALOG.
# Add one new governance policy.
# Add one new tool.
# Ask a question that requires AT LEAST TWO tool calls.
#
# Next week:
# Replace simple policy lookup with RAG over governance documents.
