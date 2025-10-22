# To run backend server
# python3 -m uvicorn backend:app --reload

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

app = FastAPI()
STATIC = Path("static")
UPLOADS = Path("uploads")
STATIC.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/")
def root():
    print("Page refreshed")
    return FileResponse(STATIC / "index.html")

@app.post("/api/upload")
async def upload(csv: UploadFile = File(...)):
    dest = UPLOADS / csv.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(csv.file, f)
    return {"ok": True, "filename": csv.filename, "saved_to": str(dest)}
