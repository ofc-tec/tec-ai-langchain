"""
10_Grading_main_v4.py

Hybrid grading-agent demo:

- ChatOllama = local agent / tool orchestration
- Google Drive API = list and download exam PDFs
- Gemini Flash = multimodal grading tool
- Solved/reference exam + rubric = grading context

The agent itself does not need vision.
It decides when to call the Gemini grading tool.
"""

from dotenv import load_dotenv
import io
import json
import os

from google import genai
from google.genai import types

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

load_dotenv()

FOLDER_ID = os.environ["GRADING_FOLDER_ID"]
CREDENTIALS_FILE = os.environ["GOOGLE_ACCOUNT_FILE"]

TOKEN_FILE = os.path.join(
    os.path.dirname(CREDENTIALS_FILE),
    "token.json",
)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Put the solved/reference exam file ID here once it is in Drive.
# Example:
# SOLVED_EXAM_FILE_ID = "1abcDEF..."
SOLVED_EXAM_FILE_ID = os.getenv("SOLVED_EXAM_FILE_ID")

# Gemini model used by the multimodal grading tool.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------
# 2. Google Drive authentication
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 3. Gemini client
# ---------------------------------------------------------------------

# google-genai reads GEMINI_API_KEY from the environment.
gemini = genai.Client()


# ---------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------

def download_drive_file(file_id: str) -> bytes:
    """
    Download a non-Google-native file from Drive and return its raw bytes.
    """

    request = drive_service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(
        buffer,
        request,
    )

    done = False

    while not done:
        _, done = downloader.next_chunk()

    return buffer.getvalue()


# ---------------------------------------------------------------------
# 5. TOOLS
# ---------------------------------------------------------------------

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

    return json.dumps(
        files,
        indent=2,
        ensure_ascii=False,
    )


@tool
def grade_exam_with_gemini(
    student_file_id: str,
    solved_exam_file_id: str,
) -> str:
    """
    Grade one student PDF exam using Gemini and a solved reference exam.

    Both files are retrieved from Google Drive.

    Returns a proposed grade with per-question feedback and a total score.
    """

    student_pdf = download_drive_file(student_file_id)
    solved_pdf = download_drive_file(solved_exam_file_id)

    prompt = """
You are grading a university exam.

You are given two PDF documents:

1. SOLVED REFERENCE EXAM
2. STUDENT EXAM

The first PDF is the solved/reference exam.
The second PDF is the student's submitted exam.

Use the solved reference as the expected mathematical solution.

GRADING RULES

- Grade strictly but fairly.
- Consider BOTH procedure and final answer.
- Give partial credit when visible work demonstrates understanding.
- Accept mathematically equivalent methods and equivalent forms.
- Do not penalize the same carried-forward error twice.
- Do not invent student work that is not visible.
- Preserve equations and mathematical notation when explaining the grade.
- Inspect all pages before grading.
- A student's supporting work may appear on a different page from the
  printed question.
- Consider plots, sketches, sequences, equations, intermediate calculations,
  and annotations when they are relevant.
- If handwriting, a calculation, or a page is unclear, explicitly say so.
- Flag uncertain cases for human review.
- The result is a PROPOSED grade. The instructor has final authority.

For every question report:

- question number
- what the student did correctly
- important errors or omissions
- proposed score
- maximum score
- short feedback
- any uncertainty that requires human review

Finally report:

- total proposed score
- maximum total score
- whether human review is recommended

Do not fabricate a rubric that is not visible in the solved/reference exam.
If point values cannot be determined from the documents, explain that clearly
instead of inventing them.
"""

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=solved_pdf,
                mime_type="application/pdf",
            ),
            types.Part.from_bytes(
                data=student_pdf,
                mime_type="application/pdf",
            ),
            prompt,
        ],
    )

    return response.text


tools = [
    list_grading_folder,
    grade_exam_with_gemini,
]


# ---------------------------------------------------------------------
# 6. Main agent
# ---------------------------------------------------------------------

# LOCAL VERSION -------------------------------------------------------
llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)

# OPENAI VERSION ------------------------------------------------------
# Easy toggle:
#
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
# )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a university grading assistant.

You have tools that can:

1. List the files in the configured Google Drive grading folder.
2. Ask Gemini to grade one student exam against a solved reference exam.

Use the tools when necessary.

IMPORTANT:

- File IDs must come from the Google Drive listing or from the user's request.
- Never invent Google Drive file IDs.
- The Gemini tool produces a proposed grade; the instructor has final authority.
- Do not claim to have visually inspected a PDF yourself.
  Gemini performs the multimodal document inspection.
- Do not grade a file until a solved/reference exam file has been identified.
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
    max_iterations=8,
    handle_parsing_errors=True,
)


# ---------------------------------------------------------------------
# 7. Run
# ---------------------------------------------------------------------

if __name__ == "__main__":

    if SOLVED_EXAM_FILE_ID:
        request = f"""
List the contents of the configured Google Drive grading folder.

The solved/reference exam has this Google Drive file ID:

{SOLVED_EXAM_FILE_ID}

Do not grade every exam yet.

For this test:
1. List the folder contents.
2. Identify the solved/reference exam.
3. Pick ONE student exam.
4. Use grade_exam_with_gemini to produce a proposed grade for that one exam.
"""
    else:
        request = """
List all contents of the configured Google Drive grading folder.

Do not grade anything yet.

Tell me that SOLVED_EXAM_FILE_ID still needs to be configured before
the Gemini grading tool can be tested.
"""

    result = executor.invoke(
        {
            "input": request
        }
    )

    print("\nGRADING AGENT RESULT")
    print("--------------------")
    print(result["output"])
