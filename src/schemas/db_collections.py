from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from core.config import settings
import logging
from core.config import settings
logger: logging.Logger = settings.LOGGING_SERVICE.logger

client = MongoClient(f"mongodb+srv://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@cluster0.fdauswj.mongodb.net/")

db = client[settings.DB_NAME]


def create_collection_if_not_exists(name, validator):
    if name in db.list_collection_names():
        logger.debug(f"Collection '{name}' already exists in database '{settings.DB_NAME}'.")
    else:
        try:
            db.create_collection(name, validator=validator)
            logger.debug(f"Created collection '{name}' with schema validation in database '{settings.DB_NAME}'.")
        except CollectionInvalid as e:
            logger.exception(f"Failed to create collection '{name}': {e}")