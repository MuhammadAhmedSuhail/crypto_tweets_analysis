# === Request Collection Schema ===
request_schema = {
    "bsonType": "object",
    "required": ["coin_name", "start_time_string", "end_time_string", "status"],
    "properties": {
        "coin_name": {"bsonType": "string"},
        "start_time_string": {"bsonType": "string"},
        "end_time_string": {"bsonType": "string"},
        "status": {"bsonType": "string"},
        "message": {"bsonType": "string"},
    }
}

request_validator = {"$jsonSchema": request_schema}

# === Ranking Collection Schema ===
ranking_schema = {
    "bsonType": "object",
    "required": ["input", "output"],
    "properties": {
        "input": {
            "bsonType": "object",
            "additionalProperties": True 
        },
        "output": {
            "bsonType": "array",
            "items": {
                "bsonType": "object",
                "additionalProperties": True
            }
        }
    }
}

ranking_validator = {"$jsonSchema": ranking_schema}