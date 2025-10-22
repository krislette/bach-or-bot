from fastapi import File, Form, HTTPException, UploadFile
from typing import Optional, Tuple
from app.utils import download_youtube_audio


# Import config values
def get_config_values():
    from app.server import MAX_FILE_SIZE, MAX_LYRICS_LENGTH, ALLOWED_AUDIO_TYPES

    return MAX_FILE_SIZE, MAX_LYRICS_LENGTH, ALLOWED_AUDIO_TYPES


def validate_lyrics(lyrics: str = Form(...)):
    """Validate lyrics length and content for multimodal endpoints."""
    _, MAX_LYRICS_LENGTH, _ = get_config_values()

    if len(lyrics) > MAX_LYRICS_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Lyrics too long. Maximum length is {MAX_LYRICS_LENGTH} characters.",
        )

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
    """Validate and process audio source from file upload or YouTube URL."""
    MAX_FILE_SIZE, _, ALLOWED_AUDIO_TYPES = get_config_values()

    if not audio_file and not youtube_url:
        raise HTTPException(
            status_code=400, detail="Either audio_file or youtube_url must be provided"
        )

    if audio_file and youtube_url:
        raise HTTPException(
            status_code=400, detail="Provide either audio_file or youtube_url, not both"
        )

    if youtube_url:
        audio_content = download_youtube_audio(youtube_url)
        return audio_content, "youtube_audio.wav", "audio/wav"

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


async def validate_audio_only(
    audio_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
) -> Tuple[Optional[bytes], str, str]:
    """Validate audio source for audio-only endpoints (no lyrics required)."""
    # Same validation as validate_audio_source but clearer naming for audio-only
    return await validate_audio_source(audio_file, youtube_url)
