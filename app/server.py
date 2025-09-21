# Fast API imports
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Model/XAI-related imports
# TODO: Import predict and predict with XAI function when available

# Utils/schemas imports
from schemas import ErrorResponse, PredictionResponse


# Initialize fast API app
app = FastAPI(title="Bach or Bot API", version="1.0.0")


# TODO: Change this to accept frontend requests from OUR OWN FE only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
async def predict_music(lyrics: str = Form(...), audio_file: UploadFile = File(...)):
    """
    Endpoint to predict whether a music sample is human-composed or AI-generated.
    """
    try:
        if audio_file.content_type not in ["audio/wav", "audio/mpeg"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "code": 1001,
                    "message": "Invalid file type. Only .wav and .mp3 are supported.",
                },
            )

        # Read the uploaded audio file
        audio_content = await audio_file.read()

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


@app.post("/api/v1/predict-xai")
async def predict_music_with_xai(
    lyrics: str = Form(...), audio_file: UploadFile = File(...)
):
    """
    Endpoint to predict whether a music sample is human-composed or AI-generated with explainability.
    """
    try:
        # Read the uploaded audio file
        audio_content = await audio_file.read()

        # TODO: Implement calling of pred + xai function/s here
        # preds = predict(audio_file)
        # xai_scores = predict_with_musiclime(audio_file)

        return JSONResponse(
            content={
                "status": "success",
                "lyrics": lyrics,
                "audio_file_name": audio_file.filename,
                "audio_content_type": audio_file.content_type,
                "audio_file_size": len(audio_content),
                # "results": preds
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
