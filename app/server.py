# Fast API imports
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Utils/schemas imports
from app.schemas import (
    ErrorResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictionXAIResponse,
    WelcomeResponse,
)
from app.utils import load_server_config, load_model_config, download_youtube_audio

# Model/XAI-related imports
from scripts.explain import musiclime
from scripts.predict import predict_pipeline

# Other imports
import io
import librosa
from typing import Optional, Tuple


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


async def validate_audio_source(
    audio_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
) -> Tuple[Optional[bytes], str, str]:
    """
    Validate and process audio source (either file or YouTube URL).
    Returns: (audio_content, file_name, content_type)
    """
    if not audio_file and not youtube_url:
        raise HTTPException(
            status_code=400, detail="Either audio_file or youtube_url must be provided"
        )

    if audio_file and youtube_url:
        raise HTTPException(
            status_code=400, detail="Provide either audio_file or youtube_url, not both"
        )

    # Process YouTube URL
    if youtube_url:
        audio_content = download_youtube_audio(youtube_url)
        return audio_content, "youtube_audio.wav", "audio/wav"

    # Process uploaded file
    if audio_file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported formats: {', '.join(ALLOWED_AUDIO_TYPES)}",
        )

    audio_content = await audio_file.read()
    if len(audio_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.",
        )

    return audio_content, audio_file.filename, audio_file.content_type


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
        results = predict_pipeline(audio_data, lyrics)

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


@app.post(
    "/api/v1/explain",
    response_model=PredictionXAIResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_music_with_xai(
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
        results = musiclime(audio_data, lyrics)

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
