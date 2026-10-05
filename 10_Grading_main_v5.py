"""
10_Grading_main_v5.py

Hybrid grading-agent demo:

- ChatOllama = local agent / tool orchestration
- Google Drive API = list and download exam PDFs
- Gemini Flash = multimodal grading tool
- Two hardcoded 30/30 reference exams:
    Version A -> DominguezCalihua_Examen1.pdf
    Version B -> ChirinoSánchez.pdf

Gemini receives:
    - solved Version A
    - solved Version B
    - one student exam

It first identifies the student's exam version, then grades against the
corresponding reference.
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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------
# 2. Hardcoded solved/reference exams
# ---------------------------------------------------------------------
#
# These were selected from the existing grading sheet because they received
# full marks (30/30), and their printed exam version was checked manually.
#

REFERENCE_EXAMS = {
    "A": {
        "name": "DominguezCalihua_Examen1.pdf",
        "file_id": "1FwAuUdeMXPbAqtF2V4W4rxn-jSH0bJZq",
        "url": (
            "https://drive.google.com/file/d/"
            "1FwAuUdeMXPbAqtF2V4W4rxn-jSH0bJZq/view"
        ),
        "score": "30/30",
    },
    "B": {
        "name": "ChirinoSánchez.pdf",
        "file_id": "1JLXok4oXtSR9STjvNF5TaayJZtV0qxsK",
        "url": (
            "https://drive.google.com/file/d/"
            "1JLXok4oXtSR9STjvNF5TaayJZtV0qxsK/view"
        ),
        "score": "30/30",
    },
}


# ---------------------------------------------------------------------
# 3. Google Drive authentication
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
# 4. Gemini client
# ---------------------------------------------------------------------

# google-genai reads GEMINI_API_KEY from the environment.
gemini = genai.Client()


# ---------------------------------------------------------------------
# 5. Helpers
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
# 6. TOOLS
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
def get_reference_exams() -> str:
    """
    Return the hardcoded solved/reference exams for Versions A and B.
    """

    return json.dumps(
        REFERENCE_EXAMS,
        indent=2,
        ensure_ascii=False,
    )


@tool
def grade_exam_with_gemini(student_file_id: str) -> str:
    """
    Grade one student PDF exam using Gemini.

    Gemini receives:
    - a solved 30/30 Version A reference,
    - a solved 30/30 Version B reference,
    - the student's exam.

    Gemini identifies the student's version and grades against the
    corresponding solved reference.
    """

    student_pdf = download_drive_file(student_file_id)

    reference_a_pdf = download_drive_file(
        REFERENCE_EXAMS["A"]["file_id"]
    )

    reference_b_pdf = download_drive_file(
        REFERENCE_EXAMS["B"]["file_id"]
    )

    prompt = """
You are grading a university Digital Signal Processing exam.

You are given THREE PDF documents in this exact order:

1. SOLVED REFERENCE EXAM — VERSION A
2. SOLVED REFERENCE EXAM — VERSION B
3. STUDENT EXAM TO GRADE

The two reference exams were previously reviewed by the instructor and each
received 30/30.

FIRST:
Identify whether the STUDENT EXAM is Version A or Version B from the printed
exam heading and/or the actual problem statements.

THEN:
Use ONLY the matching solved reference exam as the primary comparison for
grading the student exam.

The exam has three problems, each worth 10 points:

1. Even and odd components
2. Cross-correlation
3. Discrete-time Fourier series

GRADING RULES

- Grade strictly but fairly.
- Maximum total score is 30 points.
- Consider BOTH procedure and final answer.
- Give partial credit when visible work demonstrates understanding.
- Accept mathematically equivalent methods and equivalent forms.
- Do not penalize the same carried-forward error twice.
- Do not invent student work that is not visible.
- Inspect ALL pages before grading.
- Supporting work may appear on later pages or on the reverse side.
- Consider equations, sequences, plots, sketches, intermediate calculations,
  annotations, and final answers.
- Zero padding outside the support is valid where appropriate and should not
  be penalized merely for being written explicitly.
- If the student reports only part of a complete cross-correlation, grade the
  visible correct work but account for missing required lags.
- Preserve mathematical notation when explaining errors.
- If handwriting, a calculation, or a page is unclear, explicitly say so.
- Flag uncertain cases for human review.
- This is a PROPOSED grade. The instructor has final authority.

RETURN:

Exam version: A or B

For each problem:
- Problem number
- Score /10
- What was done correctly
- Important errors or omissions
- Short feedback
- Any uncertainty

Finally:
- Total score /30
- Equivalent grade /10
- Short overall comment
- Human review required: yes/no

Do not grade either solved reference exam.
Only grade the THIRD PDF, which is the student's submission.
"""

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=reference_a_pdf,
                mime_type="application/pdf",
            ),
            types.Part.from_bytes(
                data=reference_b_pdf,
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
    get_reference_exams,
    grade_exam_with_gemini,
]


# ---------------------------------------------------------------------
# 7. Main agent
# ---------------------------------------------------------------------

# LOCAL VERSION -------------------------------------------------------
llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)

# OPENAI VERSION ------------------------------------------------------
# Uncomment this and comment ChatOllama above for a faster cloud agent.
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
2. Show the two instructor-approved solved/reference exams.
3. Ask Gemini to grade ONE student exam.

The Gemini grading tool already contains both solved reference exams.
You only need to give it the Google Drive file ID of the student exam.

IMPORTANT:

- File IDs must come from the Google Drive listing or the user's request.
- Never invent Google Drive file IDs.
- Do not select either reference exam as the student exam.
- The Gemini tool produces a proposed grade; the instructor has final authority.
- Do not claim to have visually inspected a PDF yourself.
  Gemini performs the multimodal document inspection.
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
# 8. Run
# ---------------------------------------------------------------------

if __name__ == "__main__":

    request = """
List the contents of the configured Google Drive grading folder.

For this test:

1. Show which solved/reference exams are configured.
2. Pick ONE student exam that is NOT one of the references.
3. Use grade_exam_with_gemini to produce a proposed grade for that one exam.
4. Report Gemini's proposed grade.

Do not grade the entire folder yet.
"""

    result = executor.invoke(
        {
            "input": request
        }
    )

    print("\nGRADING AGENT RESULT")
    print("--------------------")
    print(result["output"])
