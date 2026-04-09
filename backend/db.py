"""
MongoDB Integration Module — db.py
For production deployment (replace JSON storage in app.py)
"""

from pymongo import MongoClient
from datetime import datetime
import os

# Connection string — set in environment variable
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "veritai"
COLLECTION = "analyses"

class Database:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION]
        # Indexes for performance
        self.collection.create_index("timestamp")
        self.collection.create_index("label")

    def insert(self, record: dict) -> str:
        """Insert a new analysis record."""
        result = self.collection.insert_one(record)
        return str(result.inserted_id)

    def get_history(self, limit: int = 50) -> list:
        """Fetch recent analyses, newest first."""
        cursor = self.collection.find({}, {'_id': 0}).sort("timestamp", -1).limit(limit)
        return list(cursor)

    def delete(self, entry_id: str) -> bool:
        """Delete a record by string ID."""
        result = self.collection.delete_one({"id": entry_id})
        return result.deleted_count > 0

    def get_stats(self) -> dict:
        """Aggregate statistics."""
        total = self.collection.count_documents({})
        fake = self.collection.count_documents({"label": "FAKE"})
        real = total - fake

        pipeline = [
            {"$unwind": "$highlighted_keywords"},
            {"$match": {"label": "FAKE"}},
            {"$group": {"_id": "$highlighted_keywords", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        top_keywords = list(self.collection.aggregate(pipeline))

        return {
            "total": total,
            "fake": fake,
            "real": real,
            "top_keywords": [{"keyword": k["_id"], "count": k["count"]} for k in top_keywords]
        }

    def clear_all(self):
        """Clear all records (admin only)."""
        self.collection.delete_many({})


# Usage example
if __name__ == "__main__":
    db = Database()
    stats = db.get_stats()
    print("DB Stats:", stats)
