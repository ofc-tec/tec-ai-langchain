"""
10_Grading_main_v6.py

Simple validation version.

Goal:
- Use ChatOllama only to call ONE grading tool.
- Grade ONE known student exam.
- Compare against two hardcoded 30/30 reference exams.
- Avoid random file selection by the agent.

Architecture:
ChatOllama -> grade_exam_with_gemini -> Drive PDFs -> Gemini Flash
"""

from dotenv import load_dotenv
import io
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

CREDENTIALS_FILE = os.environ["GOOGLE_ACCOUNT_FILE"]

TOKEN_FILE = os.path.join(
    os.path.dirname(CREDENTIALS_FILE),
    "token.json",
)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

# ---------------------------------------------------------------------
# 2. Hardcoded reference exams
# ---------------------------------------------------------------------

REFERENCE_EXAMS = {
    "A": {
        "name": "DominguezCalihua_Examen1.pdf",
        "file_id": "1FwAuUdeMXPbAqtF2V4W4rxn-jSH0bJZq",
    },
    "B": {
        "name": "ChirinoSánchez.pdf",
        "file_id": "1JLXok4oXtSR9STjvNF5TaayJZtV0qxsK",
    },
}


# ---------------------------------------------------------------------
# 3. Hardcoded student exam for validation
# ---------------------------------------------------------------------

TEST_STUDENT = {
    "name": "Tellez_Olmedo_Examen1A.pdf",
    "file_id": "1jPhHVuFiUVcDHeJW2HZYi-4dwmYUdsKy",
}


# ---------------------------------------------------------------------
# 4. Google Drive authentication
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

    return build(
        "drive",
        "v3",
        credentials=creds,
    )


drive_service = get_drive_service()


# ---------------------------------------------------------------------
# 5. Gemini client
# ---------------------------------------------------------------------

gemini = genai.Client()


# ---------------------------------------------------------------------
# 6. Helper
# ---------------------------------------------------------------------

def download_drive_file(file_id: str) -> bytes:
    request = drive_service.files().get_media(
        fileId=file_id
    )

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
# 7. Grading tool
# ---------------------------------------------------------------------

@tool
def grade_exam_with_gemini(student_file_id: str) -> str:
    """
    Grade one student exam using two solved 30/30 references.
    """

    reference_ids = {
        REFERENCE_EXAMS["A"]["file_id"],
        REFERENCE_EXAMS["B"]["file_id"],
    }

    if student_file_id in reference_ids:
        return (
            "ERROR: The selected file is a solved/reference exam. "
            "It cannot be graded as a student submission."
        )

    reference_a_pdf = download_drive_file(
        REFERENCE_EXAMS["A"]["file_id"]
    )

    reference_b_pdf = download_drive_file(
        REFERENCE_EXAMS["B"]["file_id"]
    )

    student_pdf = download_drive_file(
        student_file_id
    )

    prompt = """
You are grading a university Digital Signal Processing exam.

You are given THREE PDFs in this exact order:

1. Solved reference exam - VERSION A
2. Solved reference exam - VERSION B
3. Student exam to grade

The two solved reference exams were previously reviewed by the instructor
and each received 30/30.

FIRST:
Identify whether the student's exam is Version A or Version B.

THEN:
Grade ONLY the third PDF using the matching solved reference.

The exam contains three problems worth 10 points each:

1. Even and odd components
2. Cross-correlation
3. Discrete-time Fourier series

Rules:

- Grade strictly but fairly.
- Maximum total is 30 points.
- Focus on the final answer first, then look for redeeming points in the procedure.
- Give partial credit.
- Accept equivalent mathematical methods.
- Do not double-penalize carried-forward errors.
- Do not invent student work.
- Inspect all pages.
- Supporting work may appear on later pages.
- Consider equations, sequences, plots, sketches, calculations,
  annotations, and final answers.
- If something is unclear, say so.
- Flag uncertain cases for human review.
- This is a proposed grade only.

Return exactly this structure:

Exam version: A or B

Problem 1:
Score: X/10
Correct:
Errors:
Feedback:
Uncertainty:

Problem 2:
Score: X/10
Correct:
Errors:
Feedback:
Uncertainty:

Problem 3:
Score: X/10
Correct:
Errors:
Feedback:
Uncertainty:

Total: X/30
Equivalent grade: X/10
Overall comment:
Human review required: yes/no
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
    grade_exam_with_gemini,
]


# ---------------------------------------------------------------------
# 8. Agent
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


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a grading assistant.

For this test, your only job is to call the grading tool with the
exact Google Drive file ID provided by the user.

Do not select a different file.
Do not invent file IDs.
Do not summarize or alter Gemini's result.
Return Gemini's grading result exactly as received.
""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),
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
    handle_parsing_errors=True,
)


# ---------------------------------------------------------------------
# 9. Run one deterministic validation exam
# ---------------------------------------------------------------------

if __name__ == "__main__":

    request = f"""
Grade exactly this student exam:

Name:
{TEST_STUDENT["name"]}

Google Drive file ID:
{TEST_STUDENT["file_id"]}

Use grade_exam_with_gemini.

Do not select another student.
Return Gemini's result exactly as returned by the tool.
"""

    result = executor.invoke(
        {
            "input": request
        }
    )

    print("\nGRADING AGENT RESULT")
    print("--------------------")
    print(result["output"])
