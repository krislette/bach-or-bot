from pydantic import BaseModel
from typing import Dict, Optional


# Pydantic model for the prediction response
class PredictionResponse(BaseModel):
    status: str
    lyrics: str
    audio_file_name: str
    audio_content_type: str
    audio_file_size: int
    results: Optional[Dict] = None


class PredictionXAIResponse(BaseModel):
    status: str
    lyrics: str
    audio_file_name: str
    audio_content_type: str
    audio_file_size: int
    results: Optional[Dict] = None
    xai_scores: Optional[Dict] = None


# Pydantic model for the error response
class ErrorResponse(BaseModel):
    status: str = "error"
    code: int
    message: str
