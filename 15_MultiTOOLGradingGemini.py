"""
main.py

Finished LangChain grading-agent version before moving to MCP.

The agent itself is intentionally small:
- import tools
- choose LLM
- define prompt
- run an interactive CLI

All external capabilities live in tools.py.
"""

from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)

from tools import ALL_TOOLS


# ---------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)

# Easy cloud toggle:
#
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
# )


# ---------------------------------------------------------------------
# 2. Agent prompt
# ---------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a university Digital Signal Processing grading assistant.

You have tools for:

- listing ungraded exams,
- checking whether an exam was graded before,
- grading one exam with Gemini,
- logging a reviewed result,
- updating an existing result after instructor review,
- summarizing grading progress.

IMPORTANT RULES:

1. Never invent Google Drive file IDs.
2. Never grade one of the solved reference exams as a student submission.
3. Gemini produces a PROPOSED grade only.
4. The instructor has final authority.
5. Do not log a grade as instructor-reviewed unless the instructor explicitly
   approves or corrects it.
6. When the instructor gives a corrected grade or feedback, preserve that
   correction faithfully.
7. Prefer deterministic tools over guessing.
8. Do not silently change Gemini's grading result when reporting it.

The human instructor is part of the workflow.
""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),
    ]
)


# ---------------------------------------------------------------------
# 3. Agent
# ---------------------------------------------------------------------

agent = create_tool_calling_agent(
    llm=llm,
    tools=ALL_TOOLS,
    prompt=prompt,
)

executor = AgentExecutor(
    agent=agent,
    tools=ALL_TOOLS,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True,
)


# ---------------------------------------------------------------------
# 4. Interactive CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("\nDSP GRADING AGENT")
    print("-----------------")
    print("Examples:")
    print("  List the ungraded exams.")
    print("  Grade Tellez_Olmedo_Examen1A.pdf with file ID ...")
    print("  Log that result as 29/30 with my feedback ...")
    print("  Show me the grading summary.")
    print("\nType 'quit' to exit.\n")

    while True:

        user_input = input("Instructor: ").strip()

        if user_input.lower() in {
            "quit",
            "exit",
            "q",
        }:
            print("Goodbye.")
            break

        if not user_input:
            continue

        result = executor.invoke(
            {
                "input": user_input
            }
        )

        print("\nAgent:")
        print(result["output"])
        print()
