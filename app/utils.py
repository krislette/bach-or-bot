import io
import tempfile
import os
import yaml
import yt_dlp

from fastapi import HTTPException
from pathlib import Path
from yt_dlp.utils import DownloadError


def load_config():
    """
    Load server configs from YAML file.
    """
    # Define path first
    config_path = Path(__file__).parent.parent / "config" / "server_config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def download_youtube_audio(youtube_url: str) -> bytes:
    """
    Download audio from YouTube URL and return as bytes.
    """
    try:
        # Create a temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "audio.mp3")

            # yt-dlp options for best audio quality
            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "outtmpl": output_path.replace(".mp3", ""),
                "quiet": True,
                "no_warnings": True,
            }

            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            # Read the downloaded file
            with open(output_path, "rb") as file:
                audio_content = file.read()

            return audio_content
    except DownloadError as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download YouTube video: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing YouTube URL {str(e)}"
        )
