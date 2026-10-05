from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langchain_googledrive.tools.google_drive.tool import GoogleDriveSearchTool
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper


# ---------------------------------------------------------
# 1. Load environment
# ---------------------------------------------------------

load_dotenv()

FOLDER_ID = os.environ["GRADING_FOLDER_ID"]


# ---------------------------------------------------------
# 2. Google Drive tool
# ---------------------------------------------------------

drive_wrapper = GoogleDriveAPIWrapper(
    folder_id=FOLDER_ID,
    num_results=20,
    template="gdrive-query-in-folder",
)

drive_tool = GoogleDriveSearchTool(
    api_wrapper=drive_wrapper
)

tools = [drive_tool]


# ---------------------------------------------------------
# 3. Agent
# ---------------------------------------------------------

#llm = ChatOpenAI(
#    model="gpt-4o-mini",
#    temperature=0,
#)
llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are testing access to a Google Drive folder.

Use the Google Drive search tool to inspect the configured folder.

For this test:
- find files that appear to be exams,
- report their filenames if available,
- report their file type if available,
- briefly describe what each result appears to contain,
- do NOT grade anything yet.

Do not invent files. Only report results returned by the Google Drive tool.
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
)


# ---------------------------------------------------------
# 4. Test
# ---------------------------------------------------------

if __name__ == "__main__":

    result = executor.invoke(
        {
            "input": """
Search the configured Google Drive folder.

Find the exam files in the folder and tell me what you find.
Do not grade anything yet.
"""
        }
    )

    print("\nGOOGLE DRIVE RESULTS")
    print("--------------------")
    print(result["output"])
