"""BigQuery-backed Data Agent; build within an active MCPAdapter context.

Expected location: ~/tec_agentic_ai/agents/gcp_data.py
"""
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ID = "ia-2006"
READ_ONLY = {
    "list_dataset_ids", "get_dataset_info", "list_table_ids",
    "get_table_info", "execute_sql_readonly", "get_query_results", "get_job",
}


def build_gcp_data_agent(mcp_tools):
    """Accept discovered MCP tools and return a specialist LangChain agent."""
    tools = [t for t in mcp_tools if t.name in READ_ONLY]
    names = {t.name for t in tools}
    required = {"list_dataset_ids", "get_table_info", "execute_sql_readonly"}
    if not required.issubset(names):
        raise RuntimeError(f"Missing BigQuery tools: {required - names}")

    return create_agent(
        model=ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0),
        tools=tools,
        system_prompt=f"""You are GCP Data, a careful analytical specialist for project {PROJECT_ID}.
Use BigQuery MCP tools to inspect actual dataset/table schemas before writing SQL.
For analysis, prefer execute_sql_readonly and push aggregations (AVG, COUNT,
GROUP BY, APPROX_QUANTILES, CORR) to BigQuery. Return small, accurate
structured JSON-compatible results with labels, units, and values suitable
for the Plot Agent. LIMIT result sizes sensibly (e.g. 30-100 rows for charts).
Never request unbounded raw tables or make up numbers or schemas.
Do not modify data. Quote the executed SQL and flag unavailable data or
permission failures. Use only project {PROJECT_ID}.""",
    )
