from pymongo import MongoClient
import os

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://test-db:jzI2BiW5BzJk6mUy@cluster0.l0hcp5u.mongodb.net/?appName=Cluster0"
DB_NAME = "Leafify"

class Database:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.history_collection = self.db["predictions"]  # correct collection

    def insert_log(self, data):
        """Inserts a prediction record into the database."""
        result = self.history_collection.insert_one(data)
        return str(result.inserted_id)

    def get_all_logs(self):
        """Fetches all history logs, sorted by newest first."""
        logs = list(self.history_collection.find().sort("timestamp", -1))
        # Convert ObjectId to string for JSON serialization
        for log in logs:
            log["_id"] = str(log["_id"])
        return logs

# Create a global instance
db = Database()
