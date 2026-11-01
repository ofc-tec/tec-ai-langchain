"""
16_drive_auth.py

Google Drive OAuth authentication.
Reuses our grading agent OAuth client.

Run:
    uv run python mcp/16_drive_auth.py
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

CLIENT_FILE = Path.home() / ".config/opencode/drive-client-desktop.json"

TOKEN_FILE = (
    Path.home()
    / ".config/opencode/drive-mcp-token.json"
)


def get_credentials(interactive=False):

    creds = None

    # Reuse existing credentials
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES
        )

    # Refresh expired credentials
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            if not interactive:
                raise RuntimeError(
                    "Token refresh failed. Run 16_drive_auth.py again."
                )
            creds = None

    # Request authorization if needed
    if not creds or not creds.valid:

        if not interactive:
            raise RuntimeError(
                "Google Drive is not authenticated. "
                "Run 16_drive_auth.py first."
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(CLIENT_FILE),
            SCOPES
        )

        creds = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent"
        )

    # Store credentials
    TOKEN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    TOKEN_FILE.write_text(creds.to_json())
    TOKEN_FILE.chmod(0o600)

    return creds


if __name__ == "__main__":

    creds = get_credentials(interactive=True)

    drive = build(
        "drive",
        "v3",
        credentials=creds,
        cache_discovery=False
    )

    result = drive.files().list(
        pageSize=5,
        fields="files(id,name,mimeType)",
        q="trashed = false"
    ).execute()

    print("\nConnected to Google Drive!\n")

    for file in result.get("files", []):
        print(file["name"], file["id"])
        