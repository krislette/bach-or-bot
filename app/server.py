# Fast API imports
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Model/XAI-related imports
# TODO: Import predict and predict with XAI function when available

# Utils/schemas imports
from schemas import ErrorResponse, PredictionResponse, PredictionXAIResponse
from utils import load_config


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


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    # Map common HTTP status codes to error codes
    error_code_map = {
        400: 1001,
        422: 1002,
        500: 1003,
    }

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": error_code_map.get(exc.status_code, 9999),
            "message": str(exc.detail),
        },
    )


@app.get("/")
def root():
    """
    Root endpoint to check if the API is running.
    """
    return {
        "message": "Welcome to Bach or Bot API!",
        "endpoints": {
            "/": "This welcome message",
            "/docs": "FastAPI auto-generated API docs",
            "/api/v1/predict": "POST endpoint that returns a bach-or-bot prediction for a music sample",
            "/api/v1/predict-xai": "POST endpoint that returns a bach-or-bot prediction with explainability for a music sample",
        },
    }


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

        # TODO: Implement calling of predict function here
        # results = predict(audio_file)

        return PredictionResponse(
            status="success",
            lyrics=lyrics,
            audio_file_name=audio_file.filename,
            audio_content_type=audio_file.content_type,
            audio_file_size=len(audio_content),
            # results=preds
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/predict-xai",
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

        # TODO: Implement calling of pred + xai function/s here
        # preds = predict(audio_file)
        # xai_scores = predict_with_musiclime(audio_file)

        return PredictionXAIResponse(
            status="success",
            lyrics=lyrics,
            audio_file_name=audio_file.filename,
            audio_content_type=audio_file.content_type,
            audio_file_size=len(audio_content),
            # results=preds,
            # xai_results=xai_scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
