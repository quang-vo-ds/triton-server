from pydantic import BaseModel


class Transcript(BaseModel):
    text: str = ""
    start_time: int = 0
    end_time: int = 0
