import io
import logging
import pandas as pd
from mongoclient import req_collection, rank_collection
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from core.config import settings

logger: logging.Logger = settings.LOGGING_SERVICE.logger

req_router = APIRouter()


@req_router.get("/get_request_status")
def get_request_status(request_id: str):
    try:
        result = req_collection.find_one({"_id": request_id})

        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request ID not found")

        return {"status":result["status"], "message":result["message"]}
    except HTTPException as e:

        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.exception(f"Got an exception {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="....Ooops")

@req_router.get("/get_request_results")
def get_request_result(request_id: str):
    result_doc = rank_collection.find_one({"_id": request_id})

    if not result_doc:
        raise HTTPException(status_code=404, detail="Result not found")

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(result_doc["output"])

    # Write to in-memory CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={request_id}_results.csv"}
    )
