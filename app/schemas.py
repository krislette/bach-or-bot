from pydantic import BaseModel
from typing import Dict, List, Optional


# Pydantic model for the base response
class BaseResponse(BaseModel):
    status: str
    message: Optional[str] = None


class WelcomeResponse(BaseResponse):
    endpoints: Dict[str, str]


class ModelInfoResponse(BaseResponse):
    model_name: str
    model_version: str
    supported_formats: List[str]
    max_file_size_mb: int
    training_info: Optional[Dict] = None
    last_updated: Optional[str] = None


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


class AudioOnlyPredictionResponse(BaseModel):
    status: str
    audio_file_name: str
    audio_content_type: str
    audio_file_size: int
    results: dict


class AudioOnlyPredictionXAIResponse(BaseModel):
    status: str
    audio_file_name: str
    audio_content_type: str
    audio_file_size: int
    results: dict


class CombinedExplanationResponse(BaseModel):
    status: str
    lyrics: str
    audio_file_name: str
    audio_content_type: str
    audio_file_size: int
    results: dict  # Contains both multimodal and audio_only results


class CombinedPredictionResponse(BaseModel):
    status: str
    lyrics: str
    audio_file_name: str
    audio_content_type: str
    audio_file_size: int
    results: dict  # Contains both multimodal and audio_only predictions
