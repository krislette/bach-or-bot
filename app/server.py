# Fast API imports
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Processing imports
import librosa
import io

# Utils/schemas imports
from app.schemas import (
    ErrorResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictionXAIResponse,
    WelcomeResponse,
)
from app.utils import load_config

# Model/XAI-related imports
from scripts.explain import musiclime
from scripts.predict import predict_pipeline


# Load config at startup
config = load_config()

# Extract configuration values
MAX_FILE_SIZE = config["file_upload"]["max_file_size_mb"] * 1024 * 1024
MAX_LYRICS_LENGTH = config["file_upload"]["max_lyrics_length"]
ALLOWED_AUDIO_TYPES = config["file_upload"]["allowed_audio_types"]

# Initialize fast API app with extracted config values
app = FastAPI(title=config["server"]["title"], version=config["server"]["version"])

# Initialize CORS with config values
cors_config = config["api"]["cors"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)


async def validate_audio_file(audio_file: UploadFile = File(...)):
    """Validate audio file type and size."""
    # Check file size
    audio_content = await audio_file.read()
    if len(audio_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.",
        )

    # Check file type
    if audio_file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported formats: {', '.join(ALLOWED_AUDIO_TYPES)}",
        )

    # Reset file pointer for later use
    audio_file.file.seek(0)
    return audio_file, audio_content


def validate_lyrics(lyrics: str = Form(...)):
    """Validate lyrics length and content."""
    if len(lyrics) > MAX_LYRICS_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Lyrics too long. Maximum length is {MAX_LYRICS_LENGTH} characters.",
        )

    # Basic sanitization, remove excessive whitespace
    lyrics = lyrics.strip()
    if not lyrics:
        raise HTTPException(
            status_code=400,
            detail="Lyrics cannot be empty.",
        )

    return lyrics


@app.get("/", response_model=WelcomeResponse, tags=["Root"])
def root():
    """
    Root endpoint to check if the API is running.
    """
    return WelcomeResponse(
        status="success",
        message="Welcome to Bach or Bot API!",
        endpoints={
            "/": "This welcome message",
            "/docs": "FastAPI auto-generated API docs",
            "/api/v1/model/info": "Model information and capabilities",
            "/api/v1/predict": "POST endpoint for bach-or-bot prediction",
            "/api/v1/explain": "POST endpoint for prediction with explainability",
        },
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_music(
    lyrics: str = Depends(validate_lyrics), audio_file_data=Depends(validate_audio_file)
):
    """
    Endpoint to predict whether a music sample is human-composed or AI-generated.
    """
    try:
        # Get the audio file and content from sanitized and cleaned audio file
        audio_file, audio_content = audio_file_data

        # Load audio from uploaded file with error handling for corrupted files
        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

        # Call MLP predict runner script to get results
        results = predict_pipeline(audio_data, lyrics)

        return PredictionResponse(
            status="success",
            lyrics=lyrics,
            audio_file_name=audio_file.filename,
            audio_content_type=audio_file.content_type,
            audio_file_size=len(audio_content),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/explain",
    response_model=PredictionXAIResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_music_with_xai(
    lyrics: str = Depends(validate_lyrics), audio_file_data=Depends(validate_audio_file)
):
    """
    Endpoint to predict whether a music sample is human-composed or AI-generated with explainability.
    """
    try:
        # Get the audio file and content from sanitized and cleaned audio file
        audio_file, audio_content = audio_file_data

        # Load audio from uploaded file with error handling for corrupted files
        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

        # Call musiclime runner script to get results
        results = musiclime(audio_data, lyrics)

        return PredictionXAIResponse(
            status="success",
            lyrics=lyrics,
            audio_file_name=audio_file.filename,
            audio_content_type=audio_file.content_type,
            audio_file_size=len(audio_content),
            results=results,
        )
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

        return ModelInfoResponse(
            status="success",
            message="Model information retrieved successfully",
            model_name="Bach or Bot",
            model_version="1.0.0",  # TODO: Load from model metadata when available
            supported_formats=supported_formats,
            max_file_size_mb=config["file_upload"]["max_file_size_mb"],
            training_info={
                "dataset": "Human-Composed and AI-generated music samples",
                "architecture": "To be specified",  # TODO: Update when model is implemented
                "accuracy": "To be determined",  # TODO: Update with actual metrics
            },
            last_updated="2024-01-01T00:00:00Z",  # TODO: Update with actual timestamp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
