import json
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

token_path = Path.home() / ".config" / "earthengine" / "credentials"
with open(token_path) as f:
    creds_data = json.load(f)

creds = Credentials(
    token=creds_data.get("access_token"),
    refresh_token=creds_data.get("refresh_token"),
    token_uri="https://oauth2.googleapis.com/token",
    client_id=creds_data.get("client_id"),
    client_secret=creds_data.get("client_secret"),
    scopes=["https://www.googleapis.com/auth/drive.file"],
)
if creds.expired:
    creds.refresh(Request())

service = build("drive", "v3", credentials=creds)

file_path = Path(r"c:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\Final_Merged_with_anomalies.csv")

media = MediaFileUpload(str(file_path), mimetype="text/csv", resumable=True)
meta  = {"name": "Final_Merged_with_anomalies.csv"}
f = service.files().create(body=meta, media_body=media, fields="id, webViewLink").execute()

# Make it accessible to anyone with the link
service.permissions().create(fileId=f["id"], body={"type": "anyone", "role": "reader"}).execute()

print("Upload complete.")
print("Download link:", f"https://drive.google.com/uc?export=download&id={f['id']}")
print("View link:", f["webViewLink"])
