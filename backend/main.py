from fastapi import FastAPI, UploadFile, File
from auth import create_user, authenticate_user
from database import logs_collection
from datetime import datetime, timezone
import tempfile
import shutil

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app import process_log_file

app = FastAPI()

@app.post("/register")
def register(username: str, password: str):
    if create_user(username, password):
        return {"message": "User created"}
    return {"error": "User already exists"}

@app.post("/login")
def login(username: str, password: str):
    if authenticate_user(username, password):
        return {"message": "Login successful"}
    return {"error": "Invalid credentials"}

@app.get("/logs")
def get_logs():
    logs = list(logs_collection.find({}, {"_id": 0}))
    return {"logs": logs}

@app.get("/")
def root():
    return {"message": "Log Anomaly Detection API is running"}

@app.post("/analyze")
async def analyze_log(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # ✅ FIX: unpack tuple
        df, failed = process_log_file(temp_path)

        # build response si duhet
        result = {
            "total_rows": len(df),
            "failed_parsing": failed,
            "anomalies": int(df["anomaly"].sum()),
            "sample": df.head(50).to_dict(orient="records"),
        }

        # save to MongoDB
        logs_collection.insert_one({
            "total_rows": result["total_rows"],
            "failed_parsing": result["failed_parsing"],
            "anomalies": result["anomalies"],
            "created_at": datetime.now(timezone.utc),
            "filename": file.filename
        })

        return result

    except Exception as e:
        return {"error": str(e)}