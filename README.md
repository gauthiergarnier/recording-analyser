# recording-analyser

[![ci](https://github.com/gauthiergarnier/recording-analyser/actions/workflows/ci.yml/badge.svg)](https://github.com/gauthiergarnier/recording-analyser/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Turn an audio recording (a meeting, an interview, a phone call) into a **readable transcript with speaker labels**.

Built on top of [WhisperX](https://github.com/m-bain/whisperX) for speech recognition + word-level alignment and [pyannote-audio](https://github.com/pyannote/pyannote-audio) for speaker diarization. Runs locally — your audio never leaves your machine.

## Example output

```
[00:00:03] SPEAKER_00: Thanks everyone for joining. Let's start with the roadmap update.
[00:00:09] SPEAKER_01: Sure. We're tracking on schedule for the Q2 release, but the migration is at risk.
[00:00:18] SPEAKER_02: What's blocking it?
[00:00:21] SPEAKER_01: Mostly schema review — we should have something to show by end of week.
```

A companion `<name>.transcript.json` is written alongside the `.txt`, with full per-word timestamps and speaker IDs for downstream processing.

## What it does

1. Converts the input audio to 16 kHz mono WAV via `ffmpeg`.
2. Transcribes with Whisper (`large-v3` by default) via `faster-whisper`.
3. Force-aligns word timestamps with wav2vec2.
4. Diarizes speakers with pyannote.
5. Assigns each word to a speaker and groups consecutive segments per speaker.

Each step shows a live progress bar.

## Requirements

- macOS or Linux
- [`ffmpeg`](https://ffmpeg.org/) (audio conversion)
- Python **3.11** (3.12+ is not yet well-supported by `torch` / `pyannote.audio` / `whisperx` wheels)
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `python -m venv`
- A free [Hugging Face](https://huggingface.co) account (for the diarization model)
- ~4 GB of disk space for cached models

GPU is optional: the script auto-detects CUDA and falls back to CPU (with `int8` quantization) otherwise. On Apple Silicon it runs on CPU — accurate, but slow.

## Setup

```bash
# 1. Install system tools (macOS — use apt/dnf on Linux)
brew install ffmpeg uv

# 2. Clone and enter the repo
git clone https://github.com/gauthiergarnier/recording-analyser.git
cd recording-analyser

# 3. Create the venv and install dependencies
uv venv --python 3.11
uv pip install whisperx

# 4. Configure your Hugging Face token
cp .env.example .env
chmod 600 .env
$EDITOR .env   # paste your HF_TOKEN
```

### Hugging Face setup (one-time)

1. Create a token at <https://huggingface.co/settings/tokens> (the default *Read* scope is enough). Paste it into `.env` as `HF_TOKEN=hf_...`.
2. Visit each gated model page **while logged in** and click *Agree and access repository*:
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>

Without step 2 you'll get a 403 error during the diarization step.

## Usage

```bash
./run.sh --help                                    # show usage + flags
./run.sh ~/Downloads/meeting.m4a                   # auto-detect speaker count
./run.sh ~/Downloads/meeting.m4a --speakers 4      # known count (recommended)
./run.sh ~/Downloads/meeting.m4a --min-speakers 2 --max-speakers 5
./run.sh ~/Downloads/quick.mp3 --model medium --language en
```

Any audio format `ffmpeg` can read works (`.m4a`, `.mp3`, `.wav`, `.mp4`, `.flac`, …). Outputs land next to the input file.

### Flags

| Flag | Description |
| --- | --- |
| `audio` (positional) | Path to the input audio file. |
| `--model` | Whisper model. Default `large-v3`. Use `medium` or `small` for faster runs. |
| `--language` | ISO code (`en`, `fr`, …). Auto-detected if omitted. |
| `--speakers N` | Exact number of speakers, when known. Significantly improves diarization. |
| `--min-speakers N` / `--max-speakers N` | Bounds when the count is unknown. |
| `--batch-size N` | Whisper batch size (default 16). Lower if you hit memory issues. |

## Performance

| Hardware | `large-v3` speed | `medium` speed |
| --- | --- | --- |
| NVIDIA GPU (CUDA, fp16) | ~10–30× realtime | ~30× realtime |
| Apple Silicon / CPU (int8) | ~0.3–1× realtime | ~1–3× realtime |

A 1-hour meeting on a MacBook (M1/M2/M3) takes roughly 1–3 hours with `large-v3`, or ~20–60 minutes with `medium`. The intermediate `<name>.16k.wav` is cached, so re-runs skip step 1.

`large-v3` is noticeably better than `medium` for:
- accented English
- non-English content
- proper nouns, acronyms, and technical jargon

For a quick gist of a clean English recording, `medium` is usually fine.

## Troubleshooting

- **`401 / gated repo` during diarization** — you haven't accepted the pyannote model terms, or `HF_TOKEN` is missing/wrong.
- **`torchcodec is not installed correctly`** at startup — harmless. pyannote falls back to another decoder.
- **`Lightning automatically upgraded your loaded checkpoint`** — informational, suppressed by the script.
- **Out of memory** — drop `--batch-size`, or switch to `--model medium`.
- **`DiarizationPipeline.__init__() got an unexpected keyword argument 'use_auth_token'`** — old whisperx; upgrade with `uv pip install -U whisperx`.

## Layout

```
.
├── run.sh             Wrapper: usage help + invokes the venv's Python
├── transcribe.py      The pipeline
├── .env.example       Copy to .env and fill in your HF token
├── .gitignore
├── LICENSE            MIT
└── README.md
```

## Credits

This is just a thin pipeline glueing together excellent prior work:

- [WhisperX](https://github.com/m-bain/whisperX) by Max Bain et al.
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) by Hervé Bredin et al.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN.
- OpenAI's [Whisper](https://github.com/openai/whisper).

## License

[MIT](LICENSE).
