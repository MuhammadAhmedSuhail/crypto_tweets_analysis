from pymongo import MongoClient
from core.config import settings

client = MongoClient(f"mongodb+srv://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@cluster0.fdauswj.mongodb.net/")
db = client[settings.DB_NAME]

req_collection = db[settings.REQUEST_COLLECTION_NAME]
rank_collection = db[settings.RANKING_COLLECTION_NAME]