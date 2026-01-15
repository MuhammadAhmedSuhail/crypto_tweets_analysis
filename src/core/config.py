import os
import logging
from dotenv import load_dotenv
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic.fields import Field
from services.logging_service import LoggingService

if not load_dotenv():
    raise Exception("Could not load .env file")


class Settings(BaseSettings):
    PROJECT_TITLE: str = os.getenv("title", "COIN ANALYSIS API")
    COIN_NAME: str = os.getenv("coin_name")
    START_TIME_STRING: str = os.getenv("start_time_string")
    END_TIME_STRING: str = os.getenv("end_time_string")

    COIN_DESK_API_KEY: str = os.getenv("coin_desk_api_key")
    RAPID_API_KEY: str = os.getenv("rapid_api_key")

    AZURE_API_KEY : str = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT: str = os.getenv("AZURE_ENDPOINT")
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION")

    DATA_FOLDER: str = "../data"
    TENX_DATA_FOLDER: str = os.path.join(DATA_FOLDER, "assets/10x_coins.csv")
    NOT_TENX_DATA_FOLDER: str = os.path.join(DATA_FOLDER, "assets/10x_coins.csv")

    RUN_COIN_PRICE: bool = Field(default=True, env="RUN_COIN_PRICE")
    RUN_TWEET_RUNNER: bool = Field(default=True, env="RUN_TWEET_RUNNER")
    RUN_LLM_ANALYSIS: bool = Field(default=True, env="RUN_LLM_ANALYSIS")
    RUN_FINAL_RANKING: bool = Field(default=True, env="RUN_FINAL_RANKING")

    DB_NAME: str = os.getenv("DB_NAME")
    REQUEST_COLLECTION_NAME: str = os.getenv("REQUEST_COLLECTION_NAME")
    RANKING_COLLECTION_NAME: str = os.getenv("RANKING_COLLECTION_NAME")
    DB_USERNAME: str = os.getenv("DB_USERNAME")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    LOGGING_DIR: str = os.getenv(DATA_FOLDER, "../logs")
    LOGGING_SERVICE: Optional[logging.Logger] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if not os.path.exists(self.DATA_FOLDER):
            os.makedirs(self.DATA_FOLDER)

        if not os.path.exists(self.LOGGING_DIR):
            os.makedirs(self.LOGGING_DIR)

        self.LOGGING_SERVICE = LoggingService(log_dir=self.LOGGING_DIR)

    class Config:
        case_sensitive = True
        env_file = "./env"

settings = Settings()