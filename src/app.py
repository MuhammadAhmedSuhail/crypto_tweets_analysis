from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.base import coin_analysis_router
from core.config import settings
from schemas.db_collections import create_collection_if_not_exists
from schemas.collection_schema import ranking_validator, request_validator

logger = settings.LOGGING_SERVICE.logger

app = FastAPI(
    title=settings.PROJECT_TITLE,
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

create_collection_if_not_exists(settings.REQUEST_COLLECTION_NAME, request_validator)
create_collection_if_not_exists(settings.RANKING_COLLECTION_NAME, ranking_validator)

app.include_router(coin_analysis_router)
app.include_router(coin_analysis_router)

logger.debug("BACKEND APP STARTED")

@app.get("/")
async def ping():
    return JSONResponse(
        content={
            "status": "OK",
            "datetime": datetime.now().strftime("%d/%m/%Y, %H:%M:%S") 
        }
    )
