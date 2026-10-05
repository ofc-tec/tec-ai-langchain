from dotenv import load_dotenv
import os
import json

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent


# ---------------------------------------------------------
# 1. Load environment
# ---------------------------------------------------------

load_dotenv()

FOLDER_ID = os.environ["GRADING_FOLDER_ID"]
CREDENTIALS_FILE = os.environ["GOOGLE_ACCOUNT_FILE"]

TOKEN_FILE = os.path.join(
    os.path.dirname(CREDENTIALS_FILE),
    "token.json",
)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ---------------------------------------------------------
# 2. Google Drive authentication
# ---------------------------------------------------------

def get_drive_service():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES,
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


drive_service = get_drive_service()


# ---------------------------------------------------------
# 3. Google Drive tool
# ---------------------------------------------------------

@tool
def list_grading_folder() -> str:
    """
    List all files directly inside the configured Google Drive grading folder.

    Returns each file's name, Google Drive file ID, MIME type, and size
    when available.
    """

    response = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed = false",
        fields="files(id, name, mimeType, size)",
        pageSize=100,
        orderBy="name",
    ).execute()

    files = response.get("files", [])

    if not files:
        return "No files found in the configured grading folder."

    return json.dumps(files, indent=2, ensure_ascii=False)


tools = [list_grading_folder]


# ---------------------------------------------------------
# 4. Agent
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
You are a university grading assistant.

For this first test, your only job is to inspect the configured
Google Drive grading folder.

Use the list_grading_folder tool.

Report every file returned by the tool.
For each file, show:
- filename
- MIME type
- Google Drive file ID
- file size if available

Do not grade anything yet.
Do not invent files.
Do not omit files returned by the tool.
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
    max_iterations=4,
)


# ---------------------------------------------------------
# 5. Run
# ---------------------------------------------------------

if __name__ == "__main__":

    result = executor.invoke(
        {
            "input": """
List all contents of the configured Google Drive grading folder.
Do not grade anything yet.
"""
        }
    )

    print("\nGOOGLE DRIVE FOLDER CONTENTS")
    print("----------------------------")
    print(result["output"])
