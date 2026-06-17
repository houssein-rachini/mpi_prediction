"""
download_pdsi_precip_from_drive.py

Downloads all_82_pdsi_500m.csv and all_82_precipitation_500m.csv from
Google Drive folder 'gaul82_pdsi_precip_500m' using the Google Drive API
(same credentials as Earth Engine).

Run:
    python download_pdsi_precip_from_drive.py
"""

from __future__ import annotations

from pathlib import Path

import ee
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

FOLDER_NAME = "gaul82_pdsi_precip_500m"
FILES = [
    "all_82_pdsi_500m.csv",
    "all_82_precipitation_500m.csv",
]
OUT_DIR = Path(__file__).resolve().parent / "gee_all_vars_500m_original_ref_gaul"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service():
    """Build Drive API service using Earth Engine credentials."""
    try:
        credentials = ee.ServiceAccountCredentials(None, None)
    except Exception:
        pass

    # Use stored OAuth credentials (same as earthengine authenticate)
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    import os, json

    token_path = Path.home() / ".config" / "earthengine" / "credentials"
    if not token_path.exists():
        raise FileNotFoundError(
            "No Earth Engine credentials found. Run `earthengine authenticate` first."
        )

    with open(token_path) as f:
        creds_data = json.load(f)

    creds = Credentials(
        token=creds_data.get("access_token"),
        refresh_token=creds_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_data.get("client_id"),
        client_secret=creds_data.get("client_secret"),
        scopes=SCOPES,
    )

    if creds.expired:
        creds.refresh(Request())

    return build("drive", "v3", credentials=creds)


def find_folder_id(service, folder_name: str) -> str:
    result = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
    ).execute()
    files = result.get("files", [])
    if not files:
        raise FileNotFoundError(f"Drive folder '{folder_name}' not found.")
    return files[0]["id"]


def find_file_id(service, folder_id: str, file_name: str) -> str:
    result = service.files().list(
        q=f"name='{file_name}' and '{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
    ).execute()
    files = result.get("files", [])
    if not files:
        raise FileNotFoundError(f"File '{file_name}' not found in Drive folder.")
    return files[0]["id"]


def download_file(service, file_id: str, dest: Path) -> None:
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest.write_bytes(buf.getvalue())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to Google Drive ...")
    service = get_drive_service()

    print(f"Finding folder '{FOLDER_NAME}' ...")
    folder_id = find_folder_id(service, FOLDER_NAME)
    print(f"  Found: {folder_id}")

    for file_name in FILES:
        dest = OUT_DIR / file_name
        if dest.exists():
            print(f"  Already exists, skipping: {file_name}")
            continue
        print(f"  Downloading {file_name} ...")
        file_id = find_file_id(service, folder_id, file_name)
        download_file(service, file_id, dest)
        print(f"  Saved to {dest}")

    print("\nDone. Now run: python compute_anomalies_from_final_merged.py")


if __name__ == "__main__":
    main()
