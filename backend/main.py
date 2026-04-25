from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import tempfile
import shutil
import sys
import os

# Fix import paths for both local and Render
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jwt_utils import create_access_token, verify_token
from auth import create_user, authenticate_user
from database import logs_collection
from datetime import datetime, timezone
from utils.log_processor import process_log_file

app = FastAPI()

MAX_ANALYZE_LINES = 1_000_000
ADMIN_MAX_ANALYZE_LINES = 10_000_000

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    username = payload.get("username")
    if not username or not isinstance(username, str):
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return username


class RegisterRequest(BaseModel):
    username: str
    password: str

@app.post("/register")
def register(data: RegisterRequest):
    result = create_user(data.username, data.password)

    if result == "User created":
        return {"message": result}
    else:
        return {"error": result}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(data: LoginRequest):
    if authenticate_user(data.username, data.password):
        token = create_access_token({"username": data.username})
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    return {"error": "Invalid credentials"}

@app.delete("/logs")
def delete_logs(username: str = Depends(get_current_user)):
    logs_collection.delete_many({"username": username})
    return {"message": "Logs deleted"}

@app.get("/logs")
def get_logs(username: str = Depends(get_current_user)):
    logs = list(
        logs_collection.find(
            {"username": username},
            {"_id": 0}
        )
        .sort("created_at", -1)
        .limit(10)
    )
    return {"logs": logs}

@app.get("/")
def root():
    return {"message": "Log Anomaly Detection API is running"}

@app.post("/analyze")
async def analyze_log(
    file: UploadFile = File(...),
    max_lines: int = Form(10_000),
    username: str = Depends(get_current_user),
):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        cap = (
            ADMIN_MAX_ANALYZE_LINES
            if username.strip() == "admin"
            else MAX_ANALYZE_LINES
        )
        ml = max(1, min(int(max_lines), cap))
        df, failed = process_log_file(temp_path, max_lines=ml)

        # build response si duhet
        result = {
            "total_rows": len(df),
            "failed_parsing": failed,
            "anomalies": int(df["anomaly"].sum()),
            "sample": df.head(50).to_dict(orient="records"),
        }

        # save to MongoDB
        logs_collection.insert_one({
            "username": username,
            "total_rows": result["total_rows"],
            "failed_parsing": result["failed_parsing"],
            "anomalies": result["anomalies"],
            "created_at": datetime.now(timezone.utc),
            "filename": file.filename
        })

        return result

    except Exception as e:
        return {"error": str(e)}