# 🧠 TEC AI LangChain — ReAct Agent with Callbacks (Ollama + uv)

This repo contains a minimal **ReAct-style agent** in LangChain, designed for your Agentic AI workshops. It shows:
- A **manual ReAct loop** (no `AgentExecutor`) so students see the mechanics
- **Callbacks** for detailed tracing
- **Local LLaMA Instruct** via **Ollama** (default), with optional GPT swap
- **Reproducible installs** using **uv** with `pyproject.toml` + `uv.lock`

Repo: https://github.com/ofc-tec/tec-ai-langchain

---

## 🚀 Quick Start (clone → sync → run)

> This repository includes `pyproject.toml` and `uv.lock`, so no manual venv steps are required.

1) **Install uv**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
# irm https://astral.sh/uv/install.ps1 | iex
```

2) **Clone and install**
```bash
git clone https://github.com/ofc-tec/tec-ai-langchain.git
cd tec-ai-langchain
uv sync   # creates/manages .venv and installs exact locked versions
```

3) **Ollama (for LLaMA Instruct models)**
```bash
ollama pull llama3:instruct
# or (if you have GPU headroom):
# ollama pull llama3.1:8b-instruct
```
> Make sure the Ollama server is running (`ollama serve`) or the background service is active.

4) **Environment variables**
- Copy `.env.example` → `.env` and fill only what you use:
```dotenv
# Only for OpenAI (optional)
OPENAI_API_KEY=sk-...

# Optional for Tavily-based examples
TAVILY_API_KEY=tvly-...

# Optional if Ollama host differs
# OLLAMA_HOST=http://localhost:11434
```

5) **Run a demo**
```bash
uv run python 6_ReAct_agent_with_callbacks.py
# or another script, e.g.:
# uv run python main.py
```

You should see step-by-step ReAct traces, tool calls, and the final answer.

---

## 🧩 What’s inside (typical layout)

```
.
├── 6_ReAct_agent_with_callbacks.py   # Manual ReAct loop demo
├── callbacks.py                      # AgentCallbackHandler for tracing
├── prompt.py                         # (optional) shared ReAct templates
├── schemas.py                        # (optional) Pydantic models
├── .env.example                      # sample environment vars
├── pyproject.toml                    # uv project file
├── uv.lock                           # pinned deps for reproducibility
└── README.md
```

---

## ⚙️ Switching Models

**Default: LLaMA Instruct (Ollama)**  
```python
from langchain_ollama import ChatOllama
from callbacks import AgentCallbackHandler

llm = ChatOllama(
    model="llama3:instruct",
    temperature=0,
    callbacks=[AgentCallbackHandler()],
).bind(stop=["\nObservation:"])
```

**OpenAI (optional)**  
```python
from langchain_openai import ChatOpenAI
from callbacks import AgentCallbackHandler

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    callbacks=[AgentCallbackHandler()],
).bind(stop=["\nObservation:"])
```

---

## 🧠 ReAct Loop Notes

This example skips `AgentExecutor` so you can see the plan/act loop:

```python
agent = (
    {"input": lambda x: x["input"],
     "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])}
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)
```

At each step:
- `AgentAction` → call the tool and append `(action, observation)` to the scratchpad
- `AgentFinish` → done (final answer)

**Prompt tail tip:** end with  
`{agent_scratchpad}Thought:`  
to prevent infinite loops.

---

## 🩵 Troubleshooting

- **Infinite loop** → Ensure the prompt ends with `{agent_scratchpad}Thought:` (not `Thought: {agent_scratchpad}`) and add a `MAX_ITERS` guard.
- **Tool name mismatch (LLaMA)** → If the model writes `Action: Use the get_text_length ...`, substring/fuzzy-match the tool name before lookup.
- **Parser hiccups** → Prefer `llama3:instruct`; wrap `agent.invoke(...)` in try/except to retry with a short format hint.
- **Callbacks not showing tool events** → Add `callbacks=[AgentCallbackHandler()]` when constructing each `Tool(...)` or pass via `config` at invoke time.
- **Version conflicts** → Always run `uv sync` to honor `uv.lock`.

---

## 🪪 License / Credits

Educational scaffolding for Agentic AI (Tec). Built with **LangChain**, **Ollama**, and **uv**.
