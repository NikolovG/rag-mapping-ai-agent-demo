# uploader.py
import os

# Define where shared files are stored
UPLOAD_PATH = os.path.join(os.path.dirname(__file__), "shared_upload.csv")

def save_uploaded_csv(file_bytes: bytes) -> str:
    """
    Saves an uploaded CSV (as bytes) to disk so other files can access it.
    Returns the file path.
    """
    with open(UPLOAD_PATH, "wb") as f:
        f.write(file_bytes)
    print("File Saved!")
    return UPLOAD_PATH