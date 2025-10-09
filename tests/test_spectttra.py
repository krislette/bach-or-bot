from pathlib import Path
import time
import argparse

import torch
import torchaudio
import numpy as np

from src.utils.config_loader import RAW_DIR
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.spectttra.spectttra_trainer import (
    _init_predictor_once,
    spectttra_predict,
    spectttra_train,
)


def find_first_n_audio(raw_dir: str, n: int = 15, exts=("wav", "flac", "mp3", "m4a", "ogg")):
    p = Path(raw_dir)
    files = []
    for ext in exts:
        files.extend(list(p.rglob(f"*.{ext}")))
        if len(files) >= n:
            break
    files = sorted(files)[:n]
    if not files:
        raise FileNotFoundError(f"No audio files found in {raw_dir}")
    return files


def time_func(fn, *args, repeat=3, sync_cuda=True, **kwargs):
    for _ in range(1):
        if torch.cuda.is_available() and sync_cuda:
            torch.cuda.synchronize()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available() and sync_cuda:
            torch.cuda.synchronize()

    times = []
    last_out = out
    for _ in range(repeat):
        if torch.cuda.is_available() and sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available() and sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_out = out
    return last_out, (sum(times) / len(times))


def preprocess_with_audio_preprocessor(file_paths, target_sr=16000, max_files=None):
    """
    Use AudioPreprocessor helpers to resample/pad/normalize waveforms.
    Returns list of tensors shaped (1, samples) suitable for spectttra_train.
    """
    audio_pre = AudioPreprocessor(script="train")
    processed = []
    for p in (file_paths if max_files is None else file_paths[:max_files]):
        waveform, sr = torchaudio.load(str(p))
        # Mono
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if necessary using AudioPreprocessor helper
        waveform = audio_pre.resample_audio(original_sr=sr, waveform=waveform)
        # Ensures mono
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Pad/Trim
        waveform = audio_pre.pad_trim(waveform)
        # Normalize
        waveform = audio_pre.normalize_waveform(waveform, method=audio_pre.WAVEFORM_NORM)
        processed.append(waveform)  # shape (1, samples)
    return processed


def main(n_files: int = 15, repeat: int = 5):
    files = find_first_n_audio(RAW_DIR, n=n_files)
    print(f"[INFO] Found {len(files)} audio files, using up to {n_files}")

    # Preprocess using AudioPreprocessor helpers
    print("[PROCESS] Preprocessing audio files (resample / mono / pad / normalize)...")
    processed_waveforms = preprocess_with_audio_preprocessor(files, max_files=n_files)
    print(f"[STATUS] Preprocessed {len(processed_waveforms)} waveforms. Shape example: {processed_waveforms[0].shape}")

    # Log initializing predictor
    print("\n[PROCESS] Running _init_predictor_once()")
    print("[STATUS] Initialized predictor once")

    # Time spectttra_single on first waveform
    print("\n[PROCESS] Timing spectttra_predict() on single waveform")
    single = processed_waveforms[0]
    out_single, t_single = time_func(spectttra_predict, single, repeat=repeat)
    print(f"[RESULTS] spectttra_predict avg: {t_single:.4f}s, embedding shape: {out_single.shape}")

    # Time spectttra_batch on the batch of up to n_files
    print(f"\n[PROCESS] Timing spectttra_train() on batch of {len(processed_waveforms)} waveforms")
    out_batch, t_batch = time_func(spectttra_train, processed_waveforms, repeat=repeat)
    print(f"[RESULTS] spectttra_train avg: {t_batch:.4f}s, embeddings shape: {out_batch.shape}")

    # Quick checks
    assert out_batch.shape[0] == len(processed_waveforms), "[ERROR] Batch output not aligned with inputs"
    assert out_batch.shape[1] == out_single.shape[0], "[ERROR] Embedding dimension mismatch"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15, help="number of audio files to test (max)")
    parser.add_argument("--repeat", type=int, default=3, help="repeat count for timing averages")
    args = parser.parse_args()
    main(n_files=args.n, repeat=args.repeat)
