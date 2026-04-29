"""Transcribe + diarize an audio file with WhisperX and pyannote.

Usage:
    HF_TOKEN=hf_xxx python transcribe.py path/to/audio.m4a [--speakers N]
"""
import os

# Silence library noise BEFORE importing them.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # hides hf_hub tqdm bars
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse
import json
import logging
import subprocess
import sys
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path

warnings.filterwarnings("ignore")  # pyannote pooling std warning, torch deprecations, etc.
for name in (
    "whisperx",
    "pyannote",
    "pyannote.audio",
    "speechbrain",
    "pytorch_lightning",
    "lightning",
    "lightning.pytorch",
    "lightning.pytorch.utilities.migration",
    "lightning_fabric",
    "torio",
):
    logging.getLogger(name).setLevel(logging.ERROR)

import torch
import whisperx
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


console = Console()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def step(progress: Progress, description: str, total: float | None = None):
    """Context manager for a step. total=None → indeterminate spinner."""
    task_id = progress.add_task(description, total=total)
    try:
        yield lambda completed=None, **kw: progress.update(task_id, completed=completed, **kw) if completed is not None else progress.update(task_id, **kw)
    finally:
        # Mark complete so the bar fills out cleanly.
        if total is not None:
            progress.update(task_id, completed=total)
        else:
            progress.update(task_id, total=1, completed=1)


def ffprobe_duration(src: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        text=True,
    )
    return float(out.strip())


def to_wav_16k_mono(src: Path, progress: Progress) -> Path:
    """Convert any input to 16 kHz mono WAV (best for ASR + diarization)."""
    dst = src.with_suffix(".16k.wav")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        with step(progress, "[1/5] convert to 16 kHz mono wav (cached)", total=1) as upd:
            upd(1)
        return dst

    duration = ffprobe_duration(src)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
         "-progress", "pipe:1", "-nostats", "-loglevel", "error", str(dst)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    with step(progress, "[1/5] convert to 16 kHz mono wav", total=duration) as upd:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    secs = int(line.split("=", 1)[1]) / 1_000_000
                    upd(min(secs, duration))
                except ValueError:
                    pass
            elif line == "progress=end":
                upd(duration)
                break
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.read() if proc.stderr else ''}")
    return dst


@contextmanager
def indeterminate(progress: Progress, description: str):
    """Run an indeterminate spinner while the block executes."""
    task_id = progress.add_task(description, total=None)
    stop = threading.Event()

    def pulse():
        while not stop.wait(0.1):
            progress.update(task_id)

    t = threading.Thread(target=pulse, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        progress.update(task_id, total=1, completed=1)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--model", default="large-v3", help="whisper model size")
    ap.add_argument("--language", default=None, help="ISO code, e.g. en, fr (auto if omitted)")
    ap.add_argument("--speakers", type=int, default=None, help="known number of speakers")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        console.print("[red]ERROR:[/] set HF_TOKEN env var (and accept terms for the pyannote diarization model on HF).")
        return 2

    if not args.audio.exists():
        console.print(f"[red]ERROR:[/] not found: {args.audio}")
        return 2

    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"

    console.print(f"[dim]device={device} compute_type={compute_type} model={args.model}[/]")

    with make_progress() as progress:
        # [1/5] ffmpeg conversion (real progress)
        wav = to_wav_16k_mono(args.audio, progress)

        # [2/5] load whisper model (indeterminate — model download + load)
        with indeterminate(progress, f"[2/5] load whisper '{args.model}'"):
            asr = whisperx.load_model(args.model, device, compute_type=compute_type, language=args.language)

        # [3/5] transcribe (real % via progress_callback fired per VAD chunk)
        audio = whisperx.load_audio(str(wav))
        tx_task = progress.add_task("[3/5] transcribe", total=100)

        def tx_cb(pct: float) -> None:
            progress.update(tx_task, completed=min(max(pct, 0), 100))

        try:
            result = asr.transcribe(
                audio, batch_size=args.batch_size, language=args.language,
                progress_callback=tx_cb,
            )
        except TypeError:
            # older whisperx without progress_callback
            result = asr.transcribe(audio, batch_size=args.batch_size, language=args.language)
        progress.update(tx_task, completed=100)
        detected_lang = result["language"]
        console.print(f"[dim]      detected language: {detected_lang}[/]")

        # [4/5] align — load model (indeterminate), then run (real %)
        with indeterminate(progress, "[4/5a] load alignment model"):
            align_model, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)

        al_task = progress.add_task("[4/5b] align word timestamps", total=100)

        def al_cb(pct: float) -> None:
            progress.update(al_task, completed=min(max(pct, 0), 100))

        try:
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, device,
                return_char_alignments=False, progress_callback=al_cb,
            )
        except TypeError:
            result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
        progress.update(al_task, completed=100)

        # [5/5] diarize — has a progress_callback (0–100)
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            from whisperx import DiarizationPipeline  # type: ignore
        with indeterminate(progress, "[5/5a] load diarization model"):
            diarize = DiarizationPipeline(token=hf_token, device=device)

        diar_kwargs = {}
        if args.speakers is not None:
            diar_kwargs["num_speakers"] = args.speakers
        if args.min_speakers is not None:
            diar_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            diar_kwargs["max_speakers"] = args.max_speakers

        diar_task = progress.add_task("[5/5b] diarize speakers", total=100)

        def diar_cb(pct: float) -> None:
            progress.update(diar_task, completed=min(max(pct, 0), 100))

        try:
            diar_segments = diarize(str(wav), progress_callback=diar_cb, **diar_kwargs)
        except TypeError:
            # older whisperx: no progress_callback support
            diar_segments = diarize(str(wav), **diar_kwargs)
        progress.update(diar_task, completed=100)

        with indeterminate(progress, "       assign speakers to words"):
            result = whisperx.assign_word_speakers(diar_segments, result)

    # Write outputs next to the audio file.
    base = args.audio.with_suffix("")
    json_path = Path(f"{base}.transcript.json")
    txt_path = Path(f"{base}.transcript.txt")

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Group consecutive segments by speaker for readable output.
    lines = []
    cur_speaker = None
    cur_start = None
    cur_text: list[str] = []
    for seg in result["segments"]:
        spk = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if spk != cur_speaker:
            if cur_speaker is not None:
                lines.append(f"[{format_timestamp(cur_start)}] {cur_speaker}: {' '.join(cur_text)}")
            cur_speaker = spk
            cur_start = seg["start"]
            cur_text = [text]
        else:
            cur_text.append(text)
    if cur_speaker is not None:
        lines.append(f"[{format_timestamp(cur_start)}] {cur_speaker}: {' '.join(cur_text)}")
    txt_path.write_text("\n\n".join(lines) + "\n")

    console.print(f"\n[green]wrote:[/]\n  {txt_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
