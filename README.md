# 🧠 ReAct Agent with Callbacks (LangChain + Ollama)

This repository contains a minimal, **educational ReAct agent** implemented in LangChain.  
It demonstrates:
- **Manual ReAct execution loop** (no `AgentExecutor` required)
- **Callbacks** for detailed trace and debugging
- **Local inference** with **LLaMA Instruct** (via Ollama)
- Optional swap for **OpenAI GPT**
- Dependency and environment management via **uv** (for reproducibility)

---

## 🚀 Quick Start

### 1. Install Requirements

#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or PowerShell
# irm https://astral.sh/uv/install.ps1 | iex
```

Check:
```bash
uv --version
```

#### Create virtual environment
```bash
uv venv .venv
source .venv/bin/activate    # Linux/macOS
# or
.venv\Scripts\activate       # Windows
```

#### Install all dependencies
```bash
uv add   "langchain>=0.3.20"   "langchain-core>=0.3.20"   "langchain-community>=0.3.20"   langchain-openai   langchain-ollama   langchain-tavily==0.2.12   tavily-python   python-dotenv   pydantic>=2
```

This will create a lockfile, ensuring your students or collaborators reproduce the **exact same environment**.

---

### 2. Ollama Setup

Install Ollama: <https://ollama.com/download>

Pull an Instruct model (these follow the ReAct template better):
```bash
ollama pull llama3:instruct
# or (if you have GPU headroom)
# ollama pull llama3.1:8b-instruct
```

Ensure the server is running:
```bash
ollama serve
```

---

### 3. Environment Variables

Create a `.env` file:
```dotenv
# Only needed for OpenAI
OPENAI_API_KEY=sk-...

# Optional for Tavily
TAVILY_API_KEY=tvly-...

# Optional if Ollama isn't local
# OLLAMA_HOST=http://localhost:11434
```

---

### 4. Run

Example ReAct agent with callback tracing:
```bash
uv run python 6_ReAct_agent_with_callbacks.py
```

It will print step-by-step reasoning:
```
Hello ReAct LangChain!
AgentAction(tool='get_text_length', tool_input='"DOG"', log='...')
get_text_length enter with text='DOG'
observation=3
AgentFinish(return_values={'output': '3'}, log='Final Answer: 3')
```

---

## 🧩 Project Layout

```
.
├── 6_ReAct_agent_with_callbacks.py   # Manual ReAct loop
├── callbacks.py                      # Custom AgentCallbackHandler
├── prompt.py                         # Optional shared ReAct templates
├── schemas.py                        # Optional Pydantic response models
├── main.py                           # Example structured-agent script
├── .env.example                      # Example environment
├── pyproject.toml                    # uv project file
└── README.md
```

---

## ⚙️ Switching Models

### Use LLaMA Instruct (default)
```python
llm = ChatOllama(
    model="llama3:instruct",
    temperature=0,
    callbacks=[AgentCallbackHandler()],
).bind(stop=["\nObservation:"])
```

### Try GPT
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    callbacks=[AgentCallbackHandler()],
).bind(stop=["\nObservation:"])
```

---

## 🧠 ReAct Loop Explained

This example doesn’t use `AgentExecutor`.  
Instead, we build the loop manually:

```python
agent = (
    {"input": lambda x: x["input"],
     "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])}
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)
```

Each call to `agent.invoke()` returns:
- `AgentAction` → run the specified tool, append the `(action, observation)` to the scratchpad
- `AgentFinish` → final answer

### Loop (simplified)
```python
while not isinstance(step, AgentFinish):
    step = agent.invoke({"input": question, "agent_scratchpad": steps})
    if isinstance(step, AgentAction):
        obs = tool_map[step.tool](step.tool_input)
        steps.append((step, str(obs)))
```

---

## 🩵 Troubleshooting

| Issue | Fix |
|-------|-----|
| **Infinite loop** | Ensure prompt ends with `{agent_scratchpad}Thought:` (not `Thought: {agent_scratchpad}`). Add a `MAX_ITERS` guard. |
| **Tool not found** | LLaMA sometimes writes `Action: Use the get_text_length function ...`. Add substring/fuzzy matching in `find_tool_by_name()`. |
| **Parser errors** | Use `llama3:instruct` or handle exceptions around `agent.invoke()`. |
| **Callbacks not showing tool events** | Add `callbacks=[AgentCallbackHandler()]` inside each `Tool(...)`. |
| **Version mismatch** | Always install via `uv sync` to honor the lockfile. |

---

## 🧩 uv Workflow Cheatsheet

First-time setup:
```bash
uv venv .venv && source .venv/bin/activate
uv sync
```

Add new dependency:
```bash
uv add <package>
```

Freeze versions for students:
```bash
uv lock
```

Run reproducibly:
```bash
uv run python 6_ReAct_agent_with_callbacks.py
```

---

## 🪪 License / Credits

Built with LangChain, Ollama, and uv.  
Created for educational use in Applied Agentic AI and ReAct agent demonstrations.
