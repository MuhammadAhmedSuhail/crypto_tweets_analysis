from pydantic import BaseModel, field_validator
from datetime import datetime


class AnalysisInput(BaseModel):
    coin_name: str
    start_time_string: str
    end_time_string: str

    @field_validator('start_time_string', 'end_time_string')
    def validate_datetime_format(cls, value: str) -> str:
        try:
            datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError(f"'{value}' is not in the format 'YYYY-MM-DD HH:MM:SS'")
        return value
