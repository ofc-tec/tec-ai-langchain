"""
tools.py

Reusable tools for the DSP grading agent.

The goal is to keep the agent in main.py small and readable while all
external actions live here:

- Google Drive discovery/download
- Gemini multimodal grading
- Google Sheets grading log
- Grade lookup/update/summary

Required .env variables:

GOOGLE_ACCOUNT_FILE=/home/oscar/.credentials/credentials.json
GRADING_FOLDER_ID=...
GEMINI_API_KEY=...

Optional:
GEMINI_MODEL=gemini-3.8-flash
GRADING_SHEET_ID=1r8aH8L9G4JvD_a0aSbjSSTp9AwbnnZsZecY54APzQdU
GRADING_LOG_SHEET=Agent_Log
"""

from dotenv import load_dotenv
from datetime import datetime
import io
import json
import os
import re

from google import genai
from google.genai import types

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_core.tools import tool


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

load_dotenv()

CREDENTIALS_FILE = os.environ["GOOGLE_ACCOUNT_FILE"]
GRADING_FOLDER_ID = os.environ["GRADING_FOLDER_ID"]

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

# Existing grading spreadsheet.
GRADING_SHEET_ID = os.getenv(
    "GRADING_SHEET_ID",
    "1r8aH8L9G4JvD_a0aSbjSSTp9AwbnnZsZecY54APzQdU",
)

# We keep the agent log separate from the instructor's original table.
GRADING_LOG_SHEET = os.getenv(
    "GRADING_LOG_SHEET",
    "Agent_Log",
)

TOKEN_FILE = os.path.join(
    os.path.dirname(CREDENTIALS_FILE),
    "token.json",
)

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]


# ---------------------------------------------------------------------
# 2. Instructor-approved reference exams
# ---------------------------------------------------------------------

REFERENCE_EXAMS = {
    "A": {
        "name": "DominguezCalihua_Examen1.pdf",
        "file_id": "1FwAuUdeMXPbAqtF2V4W4rxn-jSH0bJZq",
        "score": "30/30",
    },
    "B": {
        "name": "ChirinoSánchez.pdf",
        "file_id": "1JLXok4oXtSR9STjvNF5TaayJZtV0qxsK",
        "score": "30/30",
    },
}

REFERENCE_IDS = {
    exam["file_id"]
    for exam in REFERENCE_EXAMS.values()
}


# ---------------------------------------------------------------------
# 3. Google authentication
# ---------------------------------------------------------------------

def get_google_credentials():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES,
        )

    # If the old token only had Drive permissions, force a new OAuth consent
    # so Sheets write access is added.
    if creds and not creds.has_scopes(SCOPES):
        creds = None

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

    return creds


_credentials = get_google_credentials()

drive_service = build(
    "drive",
    "v3",
    credentials=_credentials,
)

sheets_service = build(
    "sheets",
    "v4",
    credentials=_credentials,
)

gemini = genai.Client()


# ---------------------------------------------------------------------
# 4. Helpers
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


def drive_link(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def ensure_log_sheet():
    spreadsheet = sheets_service.spreadsheets().get(
        spreadsheetId=GRADING_SHEET_ID,
        fields="sheets.properties.title",
    ).execute()

    titles = {
        sheet["properties"]["title"]
        for sheet in spreadsheet.get("sheets", [])
    }

    if GRADING_LOG_SHEET not in titles:
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=GRADING_SHEET_ID,
            body={
                "requests": [
                    {
                        "addSheet": {
                            "properties": {
                                "title": GRADING_LOG_SHEET
                            }
                        }
                    }
                ]
            },
        ).execute()

    header_range = f"{GRADING_LOG_SHEET}!A1:I1"

    current = sheets_service.spreadsheets().values().get(
        spreadsheetId=GRADING_SHEET_ID,
        range=header_range,
    ).execute().get("values", [])

    if not current:
        headers = [[
            "timestamp",
            "file_name",
            "file_id",
            "file_link",
            "version",
            "proposed_grade",
            "final_grade",
            "feedback",
            "reviewed",
        ]]

        sheets_service.spreadsheets().values().update(
            spreadsheetId=GRADING_SHEET_ID,
            range=header_range,
            valueInputOption="RAW",
            body={"values": headers},
        ).execute()


def read_log_rows():
    ensure_log_sheet()

    values = sheets_service.spreadsheets().values().get(
        spreadsheetId=GRADING_SHEET_ID,
        range=f"{GRADING_LOG_SHEET}!A:I",
    ).execute().get("values", [])

    if len(values) <= 1:
        return []

    headers = values[0]

    rows = []

    for row in values[1:]:
        padded = row + [""] * (len(headers) - len(row))
        rows.append(dict(zip(headers, padded)))

    return rows


def extract_score_out_of_30(text: str):
    """
    Best-effort extraction for summaries.
    Accepts values such as:
      29/30
      26.5 / 30
    """
    if not text:
        return None

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*/\s*30",
        str(text),
    )

    if not match:
        return None

    return float(match.group(1))


# ---------------------------------------------------------------------
# 5. Tools
# ---------------------------------------------------------------------

@tool
def list_ungraded_exams() -> str:
    """
    List PDF exams in the grading folder that do not yet have a grading-log
    entry. Instructor-approved reference exams are excluded.
    """

    response = drive_service.files().list(
        q=(
            f"'{GRADING_FOLDER_ID}' in parents "
            "and trashed = false "
            "and mimeType = 'application/pdf'"
        ),
        fields="files(id,name,mimeType,size)",
        pageSize=100,
        orderBy="name",
    ).execute()

    files = response.get("files", [])

    logged_ids = {
        row.get("file_id", "")
        for row in read_log_rows()
    }

    ungraded = [
        {
            "name": f["name"],
            "file_id": f["id"],
            "file_link": drive_link(f["id"]),
        }
        for f in files
        if f["id"] not in logged_ids
        and f["id"] not in REFERENCE_IDS
    ]

    return json.dumps(
        ungraded,
        indent=2,
        ensure_ascii=False,
    )


@tool
def get_previous_grade(file_id: str) -> str:
    """
    Return the most recent grading-log entry for one Google Drive file ID.
    """

    matches = [
        row
        for row in read_log_rows()
        if row.get("file_id") == file_id
    ]

    if not matches:
        return "No previous grading record found."

    return json.dumps(
        matches[-1],
        indent=2,
        ensure_ascii=False,
    )


@tool
def grade_exam_with_gemini(student_file_id: str) -> str:
    """
    Grade one student PDF using Gemini and the two instructor-approved
    30/30 reference exams.
    """

    if student_file_id in REFERENCE_IDS:
        return (
            "ERROR: The selected file is one of the solved reference exams "
            "and cannot be graded as a student submission."
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

Both reference exams were reviewed by the instructor and received 30/30.

FIRST:
Identify whether the student's exam is Version A or Version B.

THEN:
Grade ONLY the third PDF using the matching solved reference.

The exam contains three problems worth 10 points each:

1. Even and odd components
2. Cross-correlation
3. Discrete-time Fourier series

IMPORTANT GRADING PRIORITY:

1. Check the student's FINAL RESULT for each problem first.
2. If the final result is correct, award full credit unless there is a clear
   conceptual error that invalidates that result.
3. Only when the final result is wrong or incomplete, inspect intermediate
   work, notes, calculations, annotations, plots, and sketches to recover
   justified partial credit.
4. Do not search for reasons to deduct points from an otherwise correct answer.
5. Prefer the instructor's demonstrated grading tolerance over an unnecessarily
   strict textbook-style interpretation.

GENERAL RULES:

- Grade strictly but fairly.
- Maximum total is 30 points.
- Give partial credit when appropriate.
- Accept mathematically equivalent methods and forms.
- Do not double-penalize a carried-forward error.
- Do not invent work that is not visible.
- Inspect all pages.
- Supporting work may appear on later pages.
- Explicit zero padding outside the support is valid where appropriate.
- If something is unclear, say so.
- Flag genuinely uncertain cases for human review.
- This is a proposed grade only. The instructor has final authority.

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


@tool
def log_grading_result(
    file_name: str,
    file_id: str,
    version: str,
    proposed_grade: str,
    final_grade: str,
    feedback: str,
    reviewed: bool,
) -> str:
    """
    Append one grading result to the Google Sheets Agent_Log.

    Use this after the instructor has reviewed the proposed grade.
    """

    ensure_log_sheet()

    row = [[
        datetime.now().isoformat(timespec="seconds"),
        file_name,
        file_id,
        drive_link(file_id),
        version,
        proposed_grade,
        final_grade,
        feedback,
        "yes" if reviewed else "no",
    ]]

    sheets_service.spreadsheets().values().append(
        spreadsheetId=GRADING_SHEET_ID,
        range=f"{GRADING_LOG_SHEET}!A:I",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": row},
    ).execute()

    return (
        f"Logged grading result for {file_name} "
        f"with final grade {final_grade}."
    )


@tool
def update_grading_result(
    file_id: str,
    final_grade: str,
    feedback: str,
    reviewed: bool = True,
) -> str:
    """
    Update the most recent grading-log row for one file after instructor review.
    """

    ensure_log_sheet()

    values = sheets_service.spreadsheets().values().get(
        spreadsheetId=GRADING_SHEET_ID,
        range=f"{GRADING_LOG_SHEET}!A:I",
    ).execute().get("values", [])

    if len(values) <= 1:
        return "No grading records exist yet."

    target_row = None

    # Search from bottom so the newest entry wins.
    for index in range(len(values) - 1, 0, -1):
        row = values[index]
        current_file_id = row[2] if len(row) > 2 else ""

        if current_file_id == file_id:
            target_row = index + 1
            break

    if target_row is None:
        return "No grading record found for that file ID."

    sheets_service.spreadsheets().values().update(
        spreadsheetId=GRADING_SHEET_ID,
        range=f"{GRADING_LOG_SHEET}!G{target_row}:I{target_row}",
        valueInputOption="RAW",
        body={
            "values": [[
                final_grade,
                feedback,
                "yes" if reviewed else "no",
            ]]
        },
    ).execute()

    return (
        f"Updated grading record for file ID {file_id} "
        f"to final grade {final_grade}."
    )


@tool
def get_grading_summary() -> str:
    """
    Return a small summary of the grading log.
    """

    rows = read_log_rows()

    if not rows:
        return "No grading results have been logged yet."

    reviewed = [
        row
        for row in rows
        if row.get("reviewed", "").lower() == "yes"
    ]

    scores = []

    for row in rows:
        score_text = (
            row.get("final_grade")
            or row.get("proposed_grade")
            or ""
        )

        score = extract_score_out_of_30(score_text)

        if score is not None:
            scores.append(score)

    summary = {
        "logged_exams": len(rows),
        "reviewed_exams": len(reviewed),
        "pending_review": len(rows) - len(reviewed),
    }

    if scores:
        summary["average_out_of_30"] = round(
            sum(scores) / len(scores),
            2,
        )
        summary["minimum_out_of_30"] = min(scores)
        summary["maximum_out_of_30"] = max(scores)

    return json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------
# 6. Export all tools
# ---------------------------------------------------------------------

ALL_TOOLS = [
    list_ungraded_exams,
    get_previous_grade,
    grade_exam_with_gemini,
    log_grading_result,
    update_grading_result,
    get_grading_summary,
]
