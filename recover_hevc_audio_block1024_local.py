#!/usr/bin/env python3
"""Add audio to the recovered block1024 HEVC videos.

The unknown SD-card format interleaves 1024-byte audio/private blocks between
hidden HEVC P-frame runs. After correcting the video to 20 fps, those blocks
line up at almost exactly 8000 bytes/second, which matches mono G.711 A-law:
one byte per sample at 8000 Hz.

This script extracts those 1024-byte blocks as raw A-law audio, decodes them,
optionally applies a conservative speech filter, and muxes the audio with the
already-recovered HEVC MP4s as AAC for normal MP4 player compatibility.

The final listening-approved path is:
  run_local_hevc_fps20_av8k_mild_eq_recovery.sh
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VPS_PREFIX = bytes.fromhex("0000001840010c01ffff0160")
BASE_ALLOWED = {32, 33, 34, 19, 20, 1}
GAP_ALLOWED = {0, 1}
MAX_NAL = 2 * 1024 * 1024
BLOCK_SIZE = 1024
FULL_JOA_SIZE = 256 * 1024 * 1024


def find_offsets(data: bytes, pat: bytes) -> list[int]:
    offs: list[int] = []
    pos = 0
    while True:
        hit = data.find(pat, pos)
        if hit < 0:
            return offs
        offs.append(hit)
        pos = hit + 1


def hevc_type(nal: bytes) -> int | None:
    if len(nal) < 2 or (nal[0] & 0x80):
        return None
    typ = (nal[0] >> 1) & 0x3F
    temporal_id_plus1 = nal[1] & 0x07
    layer_id = ((nal[0] & 1) << 5) | ((nal[1] >> 3) & 0x1F)
    if temporal_id_plus1 == 0 or layer_id != 0:
        return None
    return typ


def parse_nals(data: bytes, pos: int, limit: int, allowed: set[int]) -> tuple[list[tuple[int, bytes]], int]:
    nals: list[tuple[int, bytes]] = []
    while pos + 6 <= limit:
        nal_len = int.from_bytes(data[pos : pos + 4], "big")
        if nal_len <= 0 or nal_len > MAX_NAL or pos + 4 + nal_len > limit:
            break
        nal = data[pos + 4 : pos + 4 + nal_len]
        typ = hevc_type(nal)
        if typ not in allowed:
            break
        nals.append((typ, nal))
        pos += 4 + nal_len
    return nals, pos


def is_valid_base_gop(nals: list[tuple[int, bytes]]) -> bool:
    return (
        len(nals) >= 4
        and [t for t, _ in nals[:3]] == [32, 33, 34]
        and nals[3][0] in (19, 20)
    )


def extract_audio_blocks(data: bytes) -> tuple[bytes, dict[str, int]]:
    """Return raw 1024-byte A-law interleave blocks in stream order."""
    offsets = find_offsets(data, VPS_PREFIX)
    raw = bytearray()
    stats = {
        "vps_offsets": len(offsets),
        "gops": 0,
        "skipped_gops": 0,
        "video_runs": 0,
        "audio_blocks": 0,
        "audio_bytes": 0,
    }

    for idx, off in enumerate(offsets):
        limit = offsets[idx + 1] if idx + 1 < len(offsets) else len(data)
        base_nals, base_end = parse_nals(data, off, limit, BASE_ALLOWED)
        if not is_valid_base_gop(base_nals):
            stats["skipped_gops"] += 1
            continue

        stats["gops"] += 1
        gap = data[base_end:limit]
        # The gap starts with one audio block before the first hidden P-frame run.
        if len(gap) >= BLOCK_SIZE:
            raw += gap[:BLOCK_SIZE]
            stats["audio_blocks"] += 1

        pos = BLOCK_SIZE
        while pos + 6 <= len(gap):
            gap_nals, video_end = parse_nals(gap, pos, len(gap), GAP_ALLOWED)
            if not gap_nals:
                break
            stats["video_runs"] += 1
            # After every hidden P-frame run, another 1024-byte audio block follows.
            if video_end + BLOCK_SIZE <= len(gap):
                raw += gap[video_end : video_end + BLOCK_SIZE]
                stats["audio_blocks"] += 1
            pos = video_end + BLOCK_SIZE

    stats["audio_bytes"] = len(raw)
    return bytes(raw), stats


def fix_alaw_blocks(raw: bytes, mode: str) -> tuple[bytes, dict[str, int]]:
    """Experimental repair for inverted-looking A-law blocks before decoding.

    This is intentionally disabled by default. It looked plausible from byte
    distribution analysis, but listening tests on joa00001 were worse than the
    unmodified A-law stream. Keep this only for future experiments.
    """
    stats = {"fixed_blocks": 0, "mapped_samples": 0}
    if mode == "none":
        return raw, stats

    fixed = bytearray()
    for i in range(0, len(raw), BLOCK_SIZE):
        block = raw[i : i + BLOCK_SIZE]
        normal_silence = sum(x in (0xD5, 0x55) for x in block)
        inverted_silence = sum(x in (0x2A, 0xAA) for x in block)

        if mode == "autoinvert" and inverted_silence > normal_silence:
            fixed.extend((x ^ 0xFF) for x in block)
            stats["fixed_blocks"] += 1
        elif mode == "map-silence":
            for x in block:
                if x == 0x2A:
                    fixed.append(0xD5)
                    stats["mapped_samples"] += 1
                elif x == 0xAA:
                    fixed.append(0x55)
                    stats["mapped_samples"] += 1
                else:
                    fixed.append(x)
        else:
            fixed.extend(block)
    return bytes(fixed), stats


def run(cmd: list[str], log_path: Path) -> bool:
    with log_path.open("ab") as log:
        log.write(("COMMAND: " + " ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        return proc.returncode == 0


def probe(path: Path) -> str:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,sample_rate,channels,duration,nb_frames",
        "-of",
        "compact=p=0:nk=1",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip().replace("\n", " | ")


def probe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def recover_one(path: Path, video_dir: Path, out_dir: Path, args: argparse.Namespace) -> tuple[str, str]:
    if path.name.startswith("joa") and path.suffix.lower() == ".mp4" and path.stat().st_size < FULL_JOA_SIZE and not args.allow_partial:
        return "partial_copy", ""

    video_mp4 = video_dir / f"{path.stem}_block1024_hevc.mp4"
    if not video_mp4.exists():
        return "missing_video", ""

    out_mp4 = out_dir / f"{path.stem}_block1024_av.mp4"
    log_path = out_dir / f"{path.stem}_block1024_av.log"
    if out_mp4.exists() and out_mp4.stat().st_size > 1024 * 1024 and not args.overwrite:
        return "exists", probe(out_mp4)

    raw_audio, stats = extract_audio_blocks(path.read_bytes())
    if not raw_audio or stats["gops"] == 0:
        return "no_audio_blocks " + " ".join(f"{k}={v}" for k, v in stats.items()), ""

    raw_audio, fix_stats = fix_alaw_blocks(raw_audio, args.alaw_fix)
    stats.update(fix_stats)

    raw_path = out_dir / f"{path.stem}_audio.alaw"
    raw_path.write_bytes(raw_audio)
    audio_seconds = len(raw_audio) / args.input_audio_rate
    video_seconds = probe_duration(video_mp4) or 0.0

    cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-loglevel",
            "warning",
            "-f",
            args.raw_audio_format,
            "-ar",
            str(args.input_audio_rate),
            "-ac",
            "1",
            "-i",
            str(raw_path),
            "-i",
            str(video_mp4),
            "-map",
            "1:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
    ]
    if args.audio_filter:
        cmd += ["-af", args.audio_filter]
    cmd += [
            "-c:a",
            "aac",
            "-b:a",
            args.audio_bitrate,
            "-ar",
            str(args.output_audio_rate),
            "-movflags",
            "+faststart",
            str(out_mp4),
    ]
    ok = run(cmd, log_path)

    if not args.keep_raw:
        try:
            raw_path.unlink()
        except OSError:
            pass

    status = "ok" if ok and out_mp4.exists() and out_mp4.stat().st_size > 0 else "mux_failed"
    status += f" raw_format={args.raw_audio_format} input_audio_rate={args.input_audio_rate} alaw_fix={args.alaw_fix}"
    status += f" audio_seconds={audio_seconds:.3f} video_seconds={video_seconds:.3f}"
    if video_seconds:
        status += f" bytes_per_video_second={len(raw_audio) / video_seconds:.3f}"
    status += " " + " ".join(f"{k}={v}" for k, v in stats.items())
    return status, probe(out_mp4) if out_mp4.exists() else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Mux recovered block1024 HEVC video with extracted 1024-byte G.711 A-law audio blocks.")
    parser.add_argument("input_dir", nargs="?", default="recovery")
    parser.add_argument("--video-dir", help="Default: input_dir/hevc_block1024_recovery_local")
    parser.add_argument("--output", "-o", help="Default: input_dir/hevc_block1024_av_recovery_local")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--raw-audio-format", choices=("alaw", "mulaw"), default="alaw")
    parser.add_argument("--alaw-fix", choices=("none", "autoinvert", "map-silence"), default="none")
    parser.add_argument("--audio-filter", default="", help="Optional ffmpeg -af filter chain for speech enhancement")
    parser.add_argument("--input-audio-rate", type=int, default=6000)
    parser.add_argument("--output-audio-rate", type=int, default=48000)
    parser.add_argument("--audio-bitrate", default="48k")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg/ffprobe not found", file=sys.stderr)
        return 2
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else input_dir / "hevc_block1024_recovery_local"
    out_dir = Path(args.output).expanduser().resolve() if args.output else input_dir / "hevc_block1024_av_recovery_local"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("joa*.mp4"))
    if args.limit:
        files = files[: args.limit]

    summary = out_dir / "hevc_block1024_av_summary.tsv"
    with summary.open("w", encoding="utf-8") as sf:
        sf.write("source\tsize\tstatus\tmp4\tprobe\n")
        for idx, path in enumerate(files, 1):
            status, pr = recover_one(path, video_dir, out_dir, args)
            mp4 = out_dir / f"{path.stem}_block1024_av.mp4"
            sf.write(f"{path.name}\t{path.stat().st_size}\t{status}\t{mp4 if mp4.exists() else ''}\t{pr}\n")
            sf.flush()
            print(f"[{idx}/{len(files)}] {path.name}: {status}", flush=True)
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
