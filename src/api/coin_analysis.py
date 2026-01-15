import os
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
from controller import complete_pipeline
from schemas.coin_analysis import AnalysisInput
from mongoclient import req_collection, rank_collection
from core.config import settings

router = APIRouter()


# @router.post("/run-analysis")
# def run_analysis(data: AnalysisInput):
#     # Set environment variables dynamically
#     # os.environ["coin_name"] = data.coin_name
#     # os.environ["start_time_string"] = data.start_time_string
#     # os.environ["end_time_string"] = data.end_time_string

#     # Run your analysis pipeline
#     complete_pipeline()

#     return {"status": "success", "message": "Analysis completed successfully!"}


def run_pipeline_in_background(data: AnalysisInput, request_id: str):
    try:
        req_collection.update_one(
            {"_id": request_id},
            {"$set": {"status": "INPROGRESS"}}
        )

        settings.COIN_NAME = data.coin_name 
        settings.START_TIME_STRING = data.start_time_string
        settings.END_TIME_STRING = data.end_time_string

        # Run the analysis
        user_ranking_df = complete_pipeline()

        # Convert relevant columns to JSON serializable dicts
        output_data = user_ranking_df.to_dict(orient="records")

        # Insert into rank_collection
        rank_collection.insert_one({
            "_id": request_id,
            "input": {
                "coin_name": data.coin_name,
                "start_time_string": data.start_time_string,
                "end_time_string": data.end_time_string
            },
            "output": output_data
        })

        # Update main request status
        req_collection.update_one(
            {"_id": request_id},
            {"$set": {"status": "COMPLETED", "message": f"Processing request for {data.coin_name} completed."}}
        )

    except Exception as e:
        req_collection.update_one(
            {"_id": request_id},
            {"$set": {"status": "FAILED", "message": str(e)}}
        )


@router.post("/coin-analysis")
def coin_analysis(data: AnalysisInput, background_tasks: BackgroundTasks):

    # Check if the input already exists in rank_collection
    existing = rank_collection.find_one({
        "input.coin_name": data.coin_name,
        "input.start_time_string": data.start_time_string,
        "input.end_time_string": data.end_time_string
    })

    # If it exists, return the existing result
    if existing:
        return {
            "request_id": existing["_id"],
            "message": "Request already completed"
        }

    # Otherwise, generate a new request ID
    custom_id = str(uuid.uuid4())

    # Save request in req_collection
    request_data = {
        "_id": custom_id,
        "coin_name": data.coin_name,
        "start_time_string": data.start_time_string,
        "end_time_string": data.end_time_string,
        "status": "PENDING",
        "message": f"Processing request for {data.coin_name}.",
    }
    req_collection.insert_one(request_data)

    # Add background task with request_id
    background_tasks.add_task(run_pipeline_in_background, data, custom_id)

    return JSONResponse(
        content={"request_id": custom_id,
        "status": "PENDING",
        "message": "Processing started"},
        status_code=status.HTTP_202_ACCEPTED    
    )