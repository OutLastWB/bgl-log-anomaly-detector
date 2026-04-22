from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise ValueError("MONGO_URL is not set")

client = MongoClient(MONGO_URL)
db = client["log_anomaly_db"]

users_collection = db["users"]
logs_collection = db["logs"]