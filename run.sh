#!/usr/bin/env bash
# recording-analyser — wrapper around transcribe.py.
# Always invokes the venv's Python from the repo root, regardless of $PWD.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

usage() {
  cat <<'EOF'
recording-analyser — transcribe + diarize an audio file.

Usage:
  ./run.sh <audio-file> [options]
  ./run.sh -h | --help

Common options:
  --speakers N              Exact number of speakers (recommended).
  --min-speakers N          Lower bound when count is unknown.
  --max-speakers N          Upper bound when count is unknown.
  --model NAME              Whisper model: large-v3 (default), medium, small, ...
  --language CODE           ISO code (en, fr, ...). Auto-detected if omitted.
  --batch-size N            Whisper batch size (default 16).

Examples:
  ./run.sh ~/Downloads/meeting.m4a --speakers 4
  ./run.sh ~/Downloads/call.m4a --min-speakers 2 --max-speakers 5
  ./run.sh ~/Downloads/quick.mp3 --model medium --language en

Outputs (next to the input file):
  <name>.transcript.txt     Readable, grouped by speaker with timestamps.
  <name>.transcript.json    Per-word data with speaker labels.

Configuration:
  .env must contain HF_TOKEN=hf_...  (chmod 600).
  Accept terms for the gated pyannote models on huggingface.co before first run.
  See README.md for the full setup.

Full flag list (from transcribe.py):
EOF
  exec .venv/bin/python transcribe.py --help
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv not found. Run setup first (see README.md)." >&2
  exit 1
fi

exec .venv/bin/python transcribe.py "$@"
