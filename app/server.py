# Fast API imports
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Utils/schemas imports
from app.schemas import (
    ModelInfoResponse,
    PredictionResponse,
    PredictionXAIResponse,
    AudioOnlyPredictionResponse,
    AudioOnlyPredictionXAIResponse,
    WelcomeResponse,
)
from app.utils import load_server_config, load_model_config
from app.validators import validate_lyrics, validate_audio_source, validate_audio_only

# Model/XAI-related imports
from scripts.explain import musiclime_multimodal, musiclime_unimodal
from scripts.predict import predict_multimodal, predict_unimodal

# Other imports
import io
import librosa
from typing import Tuple

# Load configs at startup
server_config = load_server_config()
model_config = load_model_config()

# Extract configuration values
MAX_FILE_SIZE = server_config["file_upload"]["max_file_size_mb"] * 1024 * 1024
MAX_LYRICS_LENGTH = server_config["file_upload"]["max_lyrics_length"]
ALLOWED_AUDIO_TYPES = server_config["file_upload"]["allowed_audio_types"]

# Initialize fast API app with extracted config values
app = FastAPI(
    title=server_config["server"]["title"], version=server_config["server"]["version"]
)

# Initialize CORS with config values
cors_config = server_config["api"]["cors"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)


@app.get("/", response_model=WelcomeResponse, tags=["Root"])
def root():
    """Root endpoint to check if the API is running."""
    return WelcomeResponse(
        status="success",
        message="Welcome to Bach or Bot API!",
        endpoints={
            "/": "This welcome message",
            "/docs": "FastAPI auto-generated API docs",
            "/api/v1/model/info": "Model information and capabilities",
            "/api/v1/predict": "POST endpoint for bach-or-bot prediction (legacy)",
            "/api/v1/explain": "POST endpoint for prediction with explainability (legacy)",
            "/api/v1/predict/multimodal": "POST endpoint for multimodal prediction",
            "/api/v1/explain/multimodal": "POST endpoint for multimodal explainability",
            "/api/v1/predict/audio": "POST endpoint for audio-only prediction",
            "/api/v1/explain/audio": "POST endpoint for audio-only explainability",
        },
    )


# Legacy endpoints (backward compatibility)
@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_music_legacy(
    lyrics: str = Depends(validate_lyrics),
    audio_data_tuple: Tuple = Depends(validate_audio_source),
):
    """Legacy multimodal prediction endpoint."""
    return await predict_multimodal_endpoint(lyrics, audio_data_tuple)


@app.post("/api/v1/explain", response_model=PredictionXAIResponse)
async def explain_music_legacy(
    lyrics: str = Depends(validate_lyrics),
    audio_data_tuple: Tuple = Depends(validate_audio_source),
):
    """Legacy multimodal explanation endpoint."""
    return await explain_multimodal_endpoint(lyrics, audio_data_tuple)


# New multimodal endpoints
@app.post("/api/v1/predict/multimodal", response_model=PredictionResponse)
async def predict_multimodal_endpoint(
    lyrics: str = Depends(validate_lyrics),
    audio_data_tuple: Tuple = Depends(validate_audio_source),
):
    """
    Endpoint to predict whether a music sample is human-composed or AI-generated.
    Accepts either an audio file upload or a YouTube URL.
    """
    try:
        # Unpack validated data
        audio_content, audio_file_name, audio_content_type = audio_data_tuple

        # Load audio with librosa
        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

        # Call MLP predict runner script
        results = predict_multimodal(audio_data, lyrics)

        return PredictionResponse(
            status="success",
            lyrics=lyrics,
            audio_file_name=audio_file_name,
            audio_content_type=audio_content_type,
            audio_file_size=len(audio_content),
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/explain/multimodal", response_model=PredictionXAIResponse)
async def explain_multimodal_endpoint(
    lyrics: str = Depends(validate_lyrics),
    audio_data_tuple: Tuple = Depends(validate_audio_source),
):
    """
    Endpoint to predict whether a music sample is human-composed or AI-generated with explainability.
    Accepts either an audio file upload or a YouTube URL.
    """
    try:
        # Unpack validated data
        audio_content, audio_file_name, audio_content_type = audio_data_tuple

        # Load audio with librosa
        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

        # Call musiclime runner script
        results = musiclime_multimodal(audio_data, lyrics)

        return PredictionXAIResponse(
            status="success",
            lyrics=lyrics,
            audio_file_name=audio_file_name,
            audio_content_type=audio_content_type,
            audio_file_size=len(audio_content),
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New audio-only endpoints
@app.post("/api/v1/predict/audio", response_model=AudioOnlyPredictionResponse)
async def predict_audio_only_endpoint(
    audio_data_tuple: Tuple = Depends(validate_audio_only),
):
    """Audio-only prediction endpoint."""
    try:
        audio_content, audio_file_name, audio_content_type = audio_data_tuple

        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

        results = predict_unimodal(audio_data)

        return AudioOnlyPredictionResponse(
            status="success",
            audio_file_name=audio_file_name,
            audio_content_type=audio_content_type,
            audio_file_size=len(audio_content),
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/explain/audio", response_model=AudioOnlyPredictionXAIResponse)
async def explain_audio_only_endpoint(
    audio_data_tuple: Tuple = Depends(validate_audio_only),
):
    """Audio-only explanation endpoint."""
    try:
        audio_content, audio_file_name, audio_content_type = audio_data_tuple

        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

        results = musiclime_unimodal(audio_data, modality="audio")

        return AudioOnlyPredictionXAIResponse(
            status="success",
            audio_file_name=audio_file_name,
            audio_content_type=audio_content_type,
            audio_file_size=len(audio_content),
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the current model and its capabilities.
    """
    try:
        # Get supported formats from config
        supported_formats = [fmt.replace("audio/", "") for fmt in ALLOWED_AUDIO_TYPES]

        # Get model info from config
        model_metadata = model_config["metadata"]
        model_architecture = model_config["mlp"]

        return ModelInfoResponse(
            status="success",
            message="Model information retrieved successfully",
            model_name=model_metadata["name"],
            model_version=model_metadata["version"],
            supported_formats=supported_formats,
            max_file_size_mb=server_config["file_upload"]["max_file_size_mb"],
            training_info={
                "dataset": model_metadata["dataset"],
                "architecture": f"{model_metadata['architecture']} - Layers: {model_architecture['hidden_layers']}",
                "accuracy": model_metadata["accuracy"],
            },
            last_updated=model_metadata["last_updated"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
