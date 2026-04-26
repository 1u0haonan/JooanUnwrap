#!/usr/bin/env python3
"""Recover video from the unknown SD-card private recorder format.

Important: joa*.mp4 is not a normal MP4 container. The extension is misleading;
the file contains length-prefixed HEVC/H.265 NAL units mixed with fixed-size
audio/private blocks. This script only rebuilds the video stream. Use
recover_hevc_audio_block1024_local.py, or the final runner
run_local_hevc_fps20_av8k_mild_eq_recovery.sh, to add audio.

Observed layout inside each usable joa*.mp4:
  [4-byte big-endian length][HEVC VPS/SPS/PPS/IDR/P-slices...]
  gap area:
    1024 bytes audio/private interleave data
    one or more length-prefixed HEVC TRAIL_R/TRAIL_N P-slices
    1024 bytes audio/private interleave data
    one or more length-prefixed HEVC TRAIL_R/TRAIL_N P-slices
    ...

The fixed 1024-byte structure recovers single-slice P-frame packets and avoids
false positives left by circular overwrite gaps. The final selected recovery
uses --fps 20 so the video duration matches the 8 kHz G.711 audio stream.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VPS_PREFIX = bytes.fromhex("0000001840010c01ffff0160")
BASE_ALLOWED = {32, 33, 34, 19, 20, 1}
GAP_ALLOWED = {0, 1}  # HEVC TRAIL_N/TRAIL_R. Slice header decides P/B; samples here are P.
MAX_NAL = 2 * 1024 * 1024
FULL_JOA_SIZE = 256 * 1024 * 1024
DEFAULT_BLOCK_SIZE = 1024


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


def scan_gap_runs_block(gap: bytes, block_size: int) -> tuple[list[list[tuple[int, bytes]]], dict[str, int]]:
    """Extract predictive HEVC NAL runs separated by fixed-size audio blocks."""
    runs: list[list[tuple[int, bytes]]] = []
    stats = {
        "block_single_runs": 0,
        "block_video_bytes": 0,
        "block_trailing_bytes": 0,
    }
    # Each gap starts with one 1024-byte audio/private block. The hidden P-frame
    # run begins immediately after that block, then the pattern repeats.
    pos = block_size
    last_end = 0
    while pos + 6 <= len(gap):
        nals, end = parse_nals(gap, pos, len(gap), GAP_ALLOWED)
        if not nals:
            break
        runs.append(nals)
        if len(nals) == 1:
            stats["block_single_runs"] += 1
        stats["block_video_bytes"] += end - pos
        last_end = end
        pos = end + block_size

    if last_end:
        stats["block_trailing_bytes"] = max(0, len(gap) - last_end)
    elif len(gap) > block_size:
        stats["block_trailing_bytes"] = len(gap)
    return runs, stats


def scan_gap_runs_broad(gap: bytes, min_run: int) -> list[list[tuple[int, bytes]]]:
    """Fallback broad scan for files that do not follow the 1024-byte layout."""
    runs: list[list[tuple[int, bytes]]] = []
    pos = 0
    markers = (b"\x00\x01", b"\x02\x01")
    while pos + 6 < len(gap):
        best = -1
        for marker in markers:
            hit = gap.find(marker, pos + 4)
            if hit >= 4 and (best < 0 or hit < best):
                best = hit
        if best < 0:
            break
        candidate = best - 4
        nals, end = parse_nals(gap, candidate, len(gap), GAP_ALLOWED)
        if len(nals) >= min_run:
            runs.append(nals)
            pos = end
        else:
            pos = best + 1
    return runs


def choose_gap_runs(gap: bytes, args: argparse.Namespace) -> tuple[list[list[tuple[int, bytes]]], dict[str, int]]:
    stats = {
        "block_single_runs": 0,
        "block_video_bytes": 0,
        "block_trailing_bytes": 0,
        "broad_fallback_gaps": 0,
    }
    if args.gap_mode == "none":
        return [], stats

    block_runs: list[list[tuple[int, bytes]]] = []
    if args.gap_mode in ("block1024", "auto"):
        block_runs, block_stats = scan_gap_runs_block(gap, args.block_size)
        stats.update(block_stats)
        if args.gap_mode == "block1024" or block_runs:
            return block_runs, stats

    broad_runs = scan_gap_runs_broad(gap, args.min_gap_run)
    if broad_runs:
        stats["broad_fallback_gaps"] = 1
    return broad_runs, stats


def extract_annexb(data: bytes, args: argparse.Namespace) -> tuple[bytes, dict[str, int]]:
    offsets = find_offsets(data, VPS_PREFIX)
    raw = bytearray()
    stats = {
        "vps_offsets": len(offsets),
        "gops": 0,
        "skipped_gops": 0,
        "base_frames": 0,
        "gap_runs": 0,
        "gap_nals": 0,
        "gap_frames": 0,
        "gap_bytes": 0,
        "block_single_runs": 0,
        "block_video_bytes": 0,
        "block_trailing_bytes": 0,
        "broad_fallback_gaps": 0,
    }

    for idx, off in enumerate(offsets):
        limit = offsets[idx + 1] if idx + 1 < len(offsets) else len(data)
        base_nals, base_end = parse_nals(data, off, limit, BASE_ALLOWED)
        if not is_valid_base_gop(base_nals):
            stats["skipped_gops"] += 1
            continue

        stats["gops"] += 1
        for typ, nal in base_nals:
            raw += b"\x00\x00\x00\x01" + nal
            if typ in (0, 1, 19, 20):
                stats["base_frames"] += 1

        gap = data[base_end:limit]
        gap_runs, gap_stats = choose_gap_runs(gap, args)
        for key, value in gap_stats.items():
            stats[key] += value
        for run in gap_runs:
            stats["gap_runs"] += 1
            for typ, nal in run:
                raw += b"\x00\x00\x00\x01" + nal
                stats["gap_nals"] += 1
                stats["gap_frames"] += 1
                stats["gap_bytes"] += len(nal) + 4

    return bytes(raw), stats


def run(cmd: list[str], log_path: Path) -> bool:
    with log_path.open("ab") as log:
        log.write(("COMMAND: " + " ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        return proc.returncode == 0


def probe(path: Path, count_frames: bool = False) -> str:
    cmd = ["ffprobe", "-hide_banner", "-v", "quiet"]
    if count_frames:
        cmd += ["-count_frames"]
    cmd += [
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration,size:stream=index,codec_name,codec_type,width,height,avg_frame_rate,duration,nb_frames,nb_read_frames",
        "-of",
        "compact=p=0:nk=1",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip().replace("\n", " | ")


def candidates(input_dir: Path, include_bins: bool) -> list[Path]:
    files = sorted(input_dir.glob("joa*.mp4"))
    if include_bins:
        files += sorted(input_dir.glob("idx*.bin"))
        files += sorted(input_dir.glob("logmain*.bin"))
    return files


def recover_one(path: Path, out_dir: Path, args: argparse.Namespace) -> tuple[str, str]:
    if path.name.startswith("joa") and path.suffix.lower() == ".mp4" and path.stat().st_size < FULL_JOA_SIZE and not args.allow_partial:
        return "partial_copy", ""

    suffix = "block1024_hevc" if args.gap_mode == "block1024" and args.block_size == DEFAULT_BLOCK_SIZE else f"{args.gap_mode}_hevc"
    out_mp4 = out_dir / f"{path.stem}_{suffix}.mp4"
    log_path = out_dir / f"{path.stem}_{suffix}.log"
    if out_mp4.exists() and out_mp4.stat().st_size > 1024 * 1024 and not args.overwrite:
        return "exists", probe(out_mp4, args.count_frames)

    raw_bytes, stats = extract_annexb(path.read_bytes(), args)
    if not raw_bytes or stats["gops"] == 0:
        return "no_hevc " + " ".join(f"{k}={v}" for k, v in stats.items()), ""

    raw_path = out_dir / f"{path.stem}_{suffix}.h265"
    raw_path.write_bytes(raw_bytes)
    ok = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "hevc",
            "-r",
            args.fps,
            "-i",
            str(raw_path),
            "-c:v",
            "copy",
            "-tag:v",
            "hvc1",
            "-an",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ],
        log_path,
    )
    if not args.keep_raw:
        try:
            raw_path.unlink()
        except OSError:
            pass

    status = "ok" if ok and out_mp4.exists() and out_mp4.stat().st_size > 0 else "mux_failed"
    status += f" gap_mode={args.gap_mode} block_size={args.block_size}"
    status += " " + " ".join(f"{k}={v}" for k, v in stats.items())
    status += f" raw_bytes={len(raw_bytes)}"
    return status, probe(out_mp4, args.count_frames) if out_mp4.exists() else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover local unknown-SD HEVC clips using the 1024-byte interleave layout.")
    parser.add_argument("input_dir", nargs="?", default="recovery")
    parser.add_argument("--output", "-o", help="Default: input_dir/hevc_block1024_recovery_local")
    parser.add_argument("--include-bins", action="store_true", help="Also try idx*.bin/logmain*.bin; use --gap-mode auto for best chance")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", default="15")
    parser.add_argument("--gap-mode", choices=("block1024", "auto", "broad", "none"), default="block1024")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--min-gap-run", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--count-frames", action="store_true", help="Slower; decodes to report actual frame count")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg/ffprobe not found", file=sys.stderr)
        return 2
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve() if args.output else input_dir / "hevc_block1024_recovery_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = candidates(input_dir, args.include_bins)
    if args.limit:
        files = files[: args.limit]

    summary = out_dir / "hevc_block1024_summary.tsv"
    with summary.open("w", encoding="utf-8") as sf:
        sf.write("source\tsize\tstatus\tmp4\tprobe\n")
        for idx, path in enumerate(files, 1):
            status, pr = recover_one(path, out_dir, args)
            suffix = "block1024_hevc" if args.gap_mode == "block1024" and args.block_size == DEFAULT_BLOCK_SIZE else f"{args.gap_mode}_hevc"
            mp4 = out_dir / f"{path.stem}_{suffix}.mp4"
            sf.write(f"{path.name}\t{path.stat().st_size}\t{status}\t{mp4 if mp4.exists() else ''}\t{pr}\n")
            sf.flush()
            print(f"[{idx}/{len(files)}] {path.name}: {status}", flush=True)
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
