#!/usr/bin/env bash
# Final SD-card recovery runner selected after listening tests.
#
# Pipeline:
#   1. recover_hevc_block1024_local.py
#      - reads joa*.mp4 private recorder files
#      - extracts length-prefixed HEVC/H.265 video
#      - uses 20 fps, which matches the recovered 8 kHz audio duration
#   2. recover_hevc_audio_block1024_local.py
#      - extracts the 1024-byte interleaved audio blocks as G.711 A-law
#      - decodes them at 8000 Hz mono
#      - applies a conservative speech EQ chosen by listening tests
#      - muxes HEVC video + AAC audio into normal playable MP4 files
#
# Usage:
#   ./run_local_hevc_fps20_av8k_mild_eq_recovery.sh
#   ./run_local_hevc_fps20_av8k_mild_eq_recovery.sh recovery --limit 5
#   ./run_local_hevc_fps20_av8k_mild_eq_recovery.sh recovery --overwrite
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="recovery"
if [[ $# -gt 0 && "${1}" != --* ]]; then INPUT="$1"; shift; fi
VIDEO_OUTPUT="$INPUT/hevc_block1024_fps20_recovery_local"
if [[ $# -gt 0 && "${1}" != --* ]]; then VIDEO_OUTPUT="$1"; shift; fi
AV_OUTPUT="$INPUT/hevc_block1024_av_fps20_alaw8k_mild_eq_recovery_local"
if [[ $# -gt 0 && "${1}" != --* ]]; then AV_OUTPUT="$1"; shift; fi

# Mild, non-destructive voice EQ. Avoid heavy denoise; it made speech less clear
# in this recorder's 8 kHz G.711 source audio.
AF='highpass=f=100,lowpass=f=3600,equalizer=f=1700:t=q:w=1.2:g=3,equalizer=f=2700:t=q:w=1.0:g=2,speechnorm=e=3:c=2:r=0.0005:f=0.0005,alimiter=limit=0.95'

python3 "$SCRIPT_DIR/recover_hevc_block1024_local.py" "$INPUT" --output "$VIDEO_OUTPUT" --fps 20 "$@"
python3 "$SCRIPT_DIR/recover_hevc_audio_block1024_local.py" "$INPUT" --video-dir "$VIDEO_OUTPUT" --output "$AV_OUTPUT" --input-audio-rate 8000 --audio-bitrate 96k --audio-filter "$AF" "$@"
