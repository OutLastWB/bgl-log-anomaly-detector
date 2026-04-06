from fastapi import FastAPI, UploadFile, File
import pandas as pd
import tempfile
import shutil

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app import process_log_file  # IMPORT FROM YOUR CODE

app = FastAPI()


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

        # Process file using your existing logic
        df, failed = process_log_file(temp_path)

        # Convert results to JSON
        result = {
            "total_rows": len(df),
            "failed_parsing": failed,
            "anomalies": int(df["anomaly"].sum()),
            "sample": df.head(50).to_dict(orient="records"),
        }

        return result

    except Exception as e:
        return {"error": str(e)}