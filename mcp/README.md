# RAG → Agentic RAG → MCP

Teaching examples for the **Big Data & AI** course, in order. The goal is to understand how a local knowledge base becomes a reusable tool, and how an agent can consume either our own MCP server or a managed cloud MCP server.



## Learning path

| Example | File | Central idea |
|---|---|---|
| 8 | `8*.py` (check checkout) | Introductory RAG: retrieval supplies context to a model. |
| 9 | `9*.py` (see note below) | Ingest PDFs, TXT and CSV into Pinecone with locally generated embeddings. |
| 10 | `10_agentic_rag_pinecone.py` | Gemini chooses whether to invoke a Pinecone retrieval tool. |
| 11 | `11_mcp_rag_server.py` | Expose retrieval as an MCP tool independent of an agent. |
| 12 | `12_mcp_raw_client.py` | Barebones MCP client: discover and call a tool explicitly. |
| 13 | `13_mcp_langchain_agent.py` | LangChain/Gemini agent discovers and chooses an MCP tool. |
| 14 | `14_consume_gcp_bigquery_mcp.py` | Consume Google's **remote, managed** BigQuery MCP server. |

**Naming note:** The saved copy of example 9 is called `9_agentic_rag.py`, while its internal docstring says `9_ingest_pinecone.py`. Use the actual filename in your branch. Example 14 was originally `14_bigquery_mcp_agent.py`; rename it as suggested below and update its header.

## Architecture: three progressively different applications

**10 — Direct, agentic RAG (no MCP):**

```text
User → Gemini/LangChain agent → search_governance_knowledge()
                                   → Ollama bge-large (query embedding)
                                   → Pinecone → retrieved evidence
     ← Gemini's grounded answer
```

**11 + 12 — MCP without an LLM:**

```text
12 Raw MCP client --stdio, tools/list + tools/call--> 11 RAG MCP server
                                                       → Ollama → Pinecone
                  <-- retrieved evidence --------------┘
```

**11 + 13 — Agent uses MCP:**

```text
13 Gemini/LangChain agent → MCPAdapter --stdio--> 11 RAG MCP server
                                                → Ollama → Pinecone
                            ← retrieved evidence
          → Gemini final answer
```

**14 — Remote MCP supplied by Google:**

```text
14 LangChain agent (local Ollama llama3.1)
    → MCPAdapter + StreamableHttpTransport (HTTPS + Google OAuth)
    → https://bigquery.googleapis.com/mcp
    → BigQuery metadata tools for project ia-2006
    → grounded metadata summary
```

**Key distinction:** MCP provides standardized *tool discovery and invocation*. It does not itself create an agent. In 12, Python selects the tool. In 13 and 14, the LLM selects from tools offered to it.

## Prerequisites

Work from the repository root:

```bash
cd ~/tec_agentic_ai
git switch mcp
uv sync
```

The project environment must include its declared LangChain, MCP/FastMCP, Pinecone, Google, and Ollama dependencies. Avoid pasting secrets into source control.

For **9–13 (governance knowledge base)**:

- Run Ollama and install the `bge-large` embedding model (`ollama pull bge-large`).
- Create a **Pinecone index for externally generated vectors**, not an integrated-embedding index. Standard `bge-large` embeddings are **1024-dimensional**; configure the index accordingly (cosine similarity is suitable).
- Set `PINECONE_API_KEY` and `INDEX_NAME` in the project `.env`; examples 9 and 10 connect to an **already existing** index. Example 9 **does not create the index**.
- For Gemini-powered examples (10 and 13), configure the appropriate Google GenAI API credential and a model ID available to your account. Example 13 defaults to `GEMINI_MODEL=gemini-2.5-flash`.
- Keep the example corpus in `mcp/rag_documents/` if that is where your checked-out example 9 resolves it; verify the script's `DOCS_DIR`.

For **14 (BigQuery)**:

```bash
gcloud auth login
gcloud config set project ia-2006
ollama pull llama3.1
```

You need permission to inspect metadata in the project, and a running local Ollama service. The current example 14 uses `ChatOllama(model="llama3.1:latest")`: **Gemini credentials are not required for that version**. `gcloud auth application-default login` serves other Python/Google SDK workflows; this example specifically calls `gcloud auth print-access-token`.

## Examples 8–10: build the RAG knowledge source

### 8 — Introductory RAG

Introductory retrieval + context injection. Check `8*.py` for its exact implementation before describing individual libraries or commands.

### 9 — Ingest the governance corpus

The recovered script reads `.pdf`, `.txt`, and `.csv` documents; attaches source/page/row metadata; splits them with `RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)`; generates embeddings using `OllamaEmbeddings(model="bge-large")`; and uploads them using `PineconeVectorStore.add_documents()`.

It generates **deterministic IDs per chunk** with SHA-256 to reduce duplicate inserts when rerun. It does **not** delete stale vectors if source text changes; treat re-ingestion and cleanup as separate concerns.

```bash
# Substitute the actual filename in the checkout:
uv run python mcp/9_agentic_rag.py
```

**Important:** `INDEX_NAME` comes from `.env`. Creating document IDs is not the same as creating a Pinecone index.

### 10 — Direct Agentic RAG

`10_agentic_rag_pinecone.py` uses the **same** `bge-large` embedding model and `INDEX_NAME`, creates a retriever with `k=5`, and registers `search_governance_knowledge(query)` as a LangChain tool. Gemini decides when to use it to answer governance questions. The tool returns retrieved text with source/page/row context where available.

```bash
uv run python mcp/10_agentic_rag_pinecone.py
```

Example questions: Can face embeddings be used for student marketing? Who owns `student_gpa`? How long is it retained?

A possible student extension is **Tavily as a second tool** for external, current public evidence. A comparison between internal policy and external law may call both tools; inspect actual tool calls instead of assuming routing worked.

## Examples 11–13: expose and consume our own MCP retrieval server

### 11 — MCP server

`11_mcp_rag_server.py` exposes governance knowledge retrieval as an MCP tool (`search_governance_knowledge`). The retrieval server accesses Ollama and Pinecone; it **does not** itself generate a Gemini policy answer. The calling agent supplies reasoning, if needed.

Optional tool-discovery check:

```bash
uv run fastmcp list mcp/11_mcp_rag_server.py
```

A visible startup banner is **not required** for an MCP stdio server. Do not write ordinary debug messages to protocol stdout.

### 12 — Barebones MCP client (no agent)

Source structure (verified):

```python
server = StdioServerParameters(
    command="uv", args=["run", "python", str(SERVER_FILE)]
)
async with Client(server) as client:
    tools_result = await client.list_tools()
    result = await client.call_tool(
        "search_governance_knowledge",
        {"query": "Can face embeddings be used for student marketing?"},
    )
```

1. `StdioServerParameters` describes how to launch **11**; it does not launch it yet.
2. Entering `async with Client(server)` starts **11 as a subprocess** and initializes the MCP session over stdin/stdout.
3. `await client.list_tools()` sends **`tools/list`**. The server responds with tool names, descriptions, and schemas; the client requests them rather than passively waiting for publication.
4. `await client.call_tool(...)` sends **`tools/call`** with the explicitly chosen tool name and arguments.
5. The client prints retrieved evidence. No LLM reasoning occurs.

```bash
uv run python mcp/12_mcp_raw_client.py
```

### 13 — Agentic MCP client

Example 13 connects to the **same server 11** using:

```python
SERVER_FILE = HERE / "11_mcp_rag_server.py"
async with MCPAdapter(SERVER_FILE) as adapter:
    tools = await adapter.list_tools()
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
```

`MCPAdapter` launches server 11 as a **local stdio subprocess**, translates discovered tools for LangChain, and gives them to the Gemini agent. The agent chooses the tool and its query; when the context manager closes it normally stops the server process it started.

```bash
uv run python mcp/13_mcp_langchain_agent.py
```

**Observed successful run — October 20, 2026:** The script discovered `search_governance_knowledge`, issued `search_governance_knowledge({'query': 'face embeddings marketing ownership retention'})`, received an HTTP 200 from Ollama's `/api/embed`, and produced a source-attributed answer citing the synthetic `ai_use_policy.txt` and `student_data_catalog.csv`. This demonstrates tool discovery and execution; it is not an independent audit of those policy claims.

**Terminal usage:** Examples 12 and 13 **each start their own copy of 11**. Neither requires you to leave 11 running in a separate terminal. When these clients exit their contexts, their subprocess connections are cleaned up.

## Example 14 — Consume Google's managed BigQuery MCP

**Rename:**

```bash
git mv mcp/14_bigquery_mcp_agent.py mcp/14_consume_gcp_bigquery_mcp.py
```

Update the docstring's filename and run command after renaming.

Unlike examples 12–13, this script **does not launch a local MCP server**. It obtains a short-lived OAuth token with `gcloud auth print-access-token`, connects to `https://bigquery.googleapis.com/mcp` using `StreamableHttpTransport`, and passes an allowed subset of the discovered tools to a LangChain agent.

The current allowed tools are:

```python
ALLOWED_TOOLS = {
    "list_dataset_ids",
    "get_dataset_info",
    "list_table_ids",
    "get_table_info",
}
ENABLE_READONLY_SQL = False
```

Setting `ENABLE_READONLY_SQL = True` adds `execute_sql_readonly`, **after** metadata-only discovery is understood. Never add `execute_sql` (write-capable) for this class exercise. Filtering the tool list reduces the agent's capabilities but **does not replace BigQuery IAM permissions**.

The current code runs `ChatOllama(model="llama3.1:latest", temperature=0)` and asks it to list accessible datasets and describe up to five using tool-returned metadata only.

```bash
uv run python mcp/14_consume_gcp_bigquery_mcp.py
```

**Cleanup for example 14:** remove unused `from googleapiclient import model`, remove any outdated Gemini prerequisite, and update the run command in its header. The access token is captured without logging it; rerun the script when it expires.

## OpenCode is another MCP client

An MCP server is not the same thing as an agent. OpenCode contains its own agent runtime and can consume **both** our governance retrieval service and a separate GCP Maestro MCP wrapper.

Example project-level `opencode.json` (keep the configured model as appropriate):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "opencode/mimo-v2.6-flash-free",
  "mcp": {
    "gcp-maestro": {
      "type": "local",
      "command": ["uv", "run", "python", "agents/maestro_mcp_server.py"],
      "enabled": true,
      "timeout": 120000
    },
    "governance-rag": {
      "type": "local",
      "command": ["uv", "run", "python", "mcp/11_mcp_rag_server.py"],
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

```bash
opencode mcp list
opencode
```

OpenCode launches each **local stdio** server as a subprocess. Do not start those servers manually in separate terminals to make the configured OpenCode client work. It still requires reachable services (e.g., local Ollama) and valid credentials.

**Two distinct types of exposed capability:** `governance-rag` is a *retrieval tool*; `gcp-maestro` wraps an *agentic system* for BigQuery analysis and plots. Both can coexist, and the outer agent may call either or both. A governance answer is not an authorization mechanism: enforce access through IAM and application controls.

## Troubleshooting checklist

- **No MCP tools:** inspect paths, `uv` environment, server import errors, and `fastmcp list` output. Discovery is not proof of successful tool execution.
- **Ollama embedding error:** check that Ollama runs at `127.0.0.1:11434` and `bge-large` is installed.
- **Pinecone error:** check `PINECONE_API_KEY`, existing `INDEX_NAME`, embedding dimension/model compatibility, namespace, and network access.
- **BigQuery authentication error:** check `gcloud auth login`, active account, `ia-2006`, and BigQuery IAM. Do not print OAuth tokens.
- **LangChain beta warning:** `langchain.mcp` emitted a beta API warning during the successful example 13 run; this was not a runtime failure.
- **Questionable answer:** inspect raw retrieved evidence (12) and actual agent tool-call traces (13/14) before trusting model conclusions.

## Takeaway for the class

**8–10:** build a retriever and let an agent use it.  
**11:** expose retrieval as a reusable MCP service.  
**12:** prove MCP works without an LLM.  
**13:** put an agent back on top of the same MCP service.  
**14:** consume a managed MCP service outside our repository.

**MCP is the interface, not the intelligence.**
