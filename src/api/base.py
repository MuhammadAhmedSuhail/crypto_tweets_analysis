from fastapi import APIRouter
from api.coin_analysis import router
from api.request_details import req_router

coin_analysis_router = APIRouter()

coin_analysis_router.include_router(router, tags=["pipeline"])
coin_analysis_router.include_router(req_router, tags=["request_details"])