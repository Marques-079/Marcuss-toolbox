#!/usr/bin/env python3
"""
Resolve-friendly cumulative word reveal using MULTIPLE SRT "lanes",
then generates a REAL Final Cut Pro XML timeline (.fcpxml) that DaVinci Resolve can import.

EXTENSION ADDED:
- (Optional) Direct DaVinci Resolve scripting: create a new timeline, create video tracks (lanes),
  insert Text+ (Fusion Title) clips per SRT cue, and set the clip text programmatically.

⚠️ IMPORTANT REALITY CHECK
DaVinci Resolve’s public scripting API varies by version (18/19/20) and install.
Different builds expose slightly different method names for:
- creating timelines
- inserting generators/titles
- accessing Fusion comps inside a title
So the Resolve automation below is written with:
- multiple fallbacks
- runtime feature-detection
- clear error messages if your Resolve build exposes different names

If the “direct Text+ insertion” path isn’t available in your build, the script will fall back to
importing the generated FCPXML (which is the most reliable programmatic way to create stacked title lanes).

Requirements:
- Python 3.13 + faster-whisper + ffmpeg
- For Resolve automation: DaVinci Resolve must be installed and running,
  and your Python must be able to import DaVinciResolveScript.

On macOS, Resolve scripting module is usually at:
  /Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules

This script will try to auto-add common module paths.
"""

from __future__ import annotations

import os

# Must be set BEFORE importing huggingface_hub / faster_whisper
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import re
import sys
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Dict, Tuple, Any
import xml.etree.ElementTree as ET

from tqdm import tqdm
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel


# ============================================================
# CONFIG (edit in-script)
# ============================================================
CONFIG = {
    # Input video
    "VIDEO_PATH": "/Users/marcus/Downloads/ep2_subs.mov",

    # Output folder + prefix (script creates lane SRTs here)
    "OUT_DIR": "/Users/marcus/Downloads",
    "OUT_PREFIX": "ep2_lane",  # produces: ep2_lane_01.srt, ep2_lane_02.srt, ...

    # temp audio
    "TMP_WAV_PATH": "/Users/marcus/Downloads/__tmp_audio.wav",
    "AUDIO_SR": 16000,

    # Model
    "FW_MODEL": "small",  # tiny/base/small/medium/large-v3 OR local folder path
    "FW_DEVICE": "cpu",
    "FW_COMPUTE_TYPE": "int8",
    "HF_CACHE_DIR": "/Users/marcus/.cache/huggingface",

    # Batch behavior
    "RESET_GAP_SECONDS": 0.60,
    "MAX_WORDS_PER_BATCH": 4,
    "SPLIT_ON_PUNCTUATION": True,

    # Batch end timing
    # IMPORTANT: we force batch_end <= next_batch_start to guarantee no overlap across batches.
    "HOLD_LAST_SECONDS": 0.25,
    "CAP_HOLD_SECONDS": 2.00,

    # Minimum cue duration (Resolve sometimes hates ultra-tiny durations)
    "MIN_CUE_SECONDS": 0.08,

    # --------------------------------------------
    # FCPXML output (what Resolve wants)
    # --------------------------------------------
    "WRITE_FCPXML": True,
    "FCPXML_OUT_NAME": "ep2_lane",   # outputs: <OUT_DIR>/ep2_lane.fcpxml

    # Timeline settings (set FPS to match your timeline if possible)
    "FCPXML_VERSION": "1.8",
    "FCPXML_WIDTH": 1920,
    "FCPXML_HEIGHT": 1080,
    "FCPXML_FPS": 30,

    # If you need 29.97fps, set this instead of FPS:
    # "FCPXML_FRAME_DURATION_STR": "1001/30000s"
    "FCPXML_FRAME_DURATION_STR": None,

    # Effect UID for Basic Title (commonly used)
    "FCPXML_TITLE_EFFECT_UID": ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti",

    # Default title text styling (can be changed later inside Resolve)
    "FCPXML_FONT": "Helvetica",
    "FCPXML_FONT_SIZE": "48",
    "FCPXML_BOLD": "1",

    # --------------------------------------------
    # NEW: Direct Resolve Automation (optional)
    # --------------------------------------------
    # If True, the script will try to:
    #  - connect to Resolve
    #  - create/choose project
    #  - create an empty timeline
    #  - ensure enough video tracks for lanes
    #  - insert Text+ (Fusion Title) per SRT cue & set its text
    #
    # If your Resolve build doesn’t expose the needed insertion calls,
    # it will fall back to importing the FCPXML it generated.
    "APPLY_TO_RESOLVE": True,

    # Which approach to try first:
    #  - "text_plus_from_srts": create tracks + insert Text+ using lane SRTs (best match to your ask)
    #  - "import_fcpxml": just import the generated FCPXML (most reliable)
    "RESOLVE_MODE": "text_plus_from_srts",

    # Project / timeline naming
    "RESOLVE_PROJECT_NAME": "Lane Import Project",
    "RESOLVE_TIMELINE_NAME": "Lane Titles (Text+)",

    # The Title name to insert (Effects Library -> Titles). Common candidates:
    # "Text+" or "Text Plus" depending on locale/version.
    "RESOLVE_TITLE_NAME": "Text+",

    # If we manage to insert a Fusion title, we’ll look for these tool names to set StyledText
    "RESOLVE_FUSION_TEXT_TOOL_CANDIDATES": ["Text1", "Text+ 1", "TextPlus1", "TextPlus", "Title Text", "Text"],

    # If you want lane 1 to be top track or bottom track:
    # "lane_1_is_top": True means lane 1 maps to highest video track index.
    # (Resolve track indices usually start at 1 and grow upward; "top" is highest index)
    "RESOLVE_LANE_1_IS_TOP": False,

    # Sometimes you want a small pad to avoid 1-frame flicker at cuts
    "RESOLVE_PAD_START_FRAMES": 0,
    "RESOLVE_PAD_END_FRAMES": 0,
}
# ============================================================


# -----------------------------
# Utilities
# -----------------------------
def require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Missing dependency: {name} not found in PATH.\n"
            f"On macOS: brew install ffmpeg"
        )


def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stderr.strip()}")
    return p.stdout.strip()


def ffprobe_duration_seconds(video_path: str) -> float:
    out = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ])
    return float(out)


def extract_audio_wav(video_path: str, wav_path: str, sr: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        wav_path
    ])


def s_to_tc(s: float) -> str:
    if s < 0:
        s = 0.0
    ms_total = int(round(s * 1000.0))
    hh = ms_total // 3_600_000
    ms_total %= 3_600_000
    mm = ms_total // 60_000
    ms_total %= 60_000
    ss = ms_total // 1000
    ms = ms_total % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def clean_word(w: str) -> str:
    return (w or "").strip().replace("\u200b", "")


def ends_sentence(w: str) -> bool:
    return bool(re.search(r"[.!?]$", (w or "").strip()))


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class WordTS:
    word: str
    start: float
    end: float


@dataclass
class Batch:
    start: float
    end: float
    words: List[WordTS]


# -----------------------------
# Model download (no Xet)
# -----------------------------
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def ensure_model_local(model_name_or_path: str, cache_dir: str) -> str:
    if os.path.exists(model_name_or_path):
        return os.path.abspath(model_name_or_path)

    key = model_name_or_path.strip().lower()
    if key not in MODEL_REPOS:
        raise ValueError(f"Unknown FW_MODEL '{model_name_or_path}'. Use {sorted(MODEL_REPOS.keys())} or a local path.")

    repo_id = MODEL_REPOS[key]
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[model] Downloading: {repo_id} (Xet disabled)")
    local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    return local_dir


# -----------------------------
# Transcription
# -----------------------------
def faster_whisper_words(wav_path: str) -> List[WordTS]:
    model_local = ensure_model_local(str(CONFIG["FW_MODEL"]), str(CONFIG["HF_CACHE_DIR"]))

    print("[1/2] Loading faster-whisper model...")
    model = WhisperModel(
        model_local,
        device=str(CONFIG["FW_DEVICE"]),
        compute_type=str(CONFIG["FW_COMPUTE_TYPE"]),
    )

    print("[2/2] Transcribing with word timestamps...")
    segments, _info = model.transcribe(
        wav_path,
        word_timestamps=True,
        vad_filter=True,
    )

    seg_list = list(segments)
    words: List[WordTS] = []

    for seg in tqdm(seg_list, desc="Reading segments", total=len(seg_list), leave=True):
        for w in (getattr(seg, "words", None) or []):
            ww = clean_word(getattr(w, "word", ""))
            if not ww:
                continue
            st = float(getattr(w, "start", 0.0))
            en = float(getattr(w, "end", st + 0.02))
            if en <= st:
                en = st + 0.02
            words.append(WordTS(ww, st, en))

    words.sort(key=lambda x: x.start)
    return words


# -----------------------------
# Batching (forces Resolve-friendly ends)
# -----------------------------
def make_batches(
    words: List[WordTS],
    video_end: float,
    reset_gap: float,
    max_words: int,
    hold_last: float,
    cap_hold: float,
    split_on_punct: bool,
    min_cue: float,
) -> List[Batch]:
    if not words:
        return []

    batches_raw: List[List[WordTS]] = []
    cur: List[WordTS] = []
    last_end: Optional[float] = None

    for w in tqdm(words, desc="Batching words", total=len(words), leave=True):
        gap = None if last_end is None else (w.start - last_end)

        new_batch = False
        if cur and gap is not None and gap >= reset_gap:
            new_batch = True
        if cur and len(cur) >= max_words:
            new_batch = True

        if new_batch and cur:
            batches_raw.append(cur)
            cur = []

        cur.append(w)
        last_end = w.end

        if split_on_punct and ends_sentence(w.word) and cur:
            batches_raw.append(cur)
            cur = []
            last_end = None

    if cur:
        batches_raw.append(cur)

    out: List[Batch] = []
    for i, b in enumerate(batches_raw):
        if not b:
            continue

        batch_start = b[0].start
        last_word_end = b[-1].end

        next_batch_start = None
        if i + 1 < len(batches_raw) and batches_raw[i + 1]:
            next_batch_start = batches_raw[i + 1][0].start

        if next_batch_start is not None:
            candidate_end = min(next_batch_start, last_word_end + cap_hold)
            batch_end = min(candidate_end, next_batch_start)  # no overlap across batches
        else:
            batch_end = min(last_word_end + hold_last, last_word_end + cap_hold)

        batch_end = min(batch_end, video_end)

        if batch_end <= batch_start:
            batch_end = min(video_end, batch_start + min_cue)

        out.append(Batch(start=batch_start, end=batch_end, words=b))

    return out


# -----------------------------
# Write MULTI-LANE SRTs
# -----------------------------
def write_lane_srts(batches: List[Batch], out_dir: str, prefix: str, min_cue: float) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    max_lane = max((len(b.words) for b in batches), default=0)
    if max_lane == 0:
        raise RuntimeError("No words found, nothing to write.")

    lane_paths: List[str] = []
    lane_files: Dict[int, Any] = {}

    try:
        for lane in range(1, max_lane + 1):
            path = os.path.join(out_dir, f"{prefix}_{lane:02d}.srt")
            lane_paths.append(path)
            lane_files[lane] = open(path, "w", encoding="utf-8")

        lane_idx: Dict[int, int] = {lane: 1 for lane in range(1, max_lane + 1)}

        for b in tqdm(batches, desc="Writing lane SRTs", total=len(batches), leave=True):
            for lane, word in enumerate(b.words, start=1):
                f = lane_files[lane]

                start = float(word.start)
                end = float(b.end)

                if end - start < min_cue:
                    end = start + min_cue

                idx = lane_idx[lane]
                f.write(f"{idx}\n")
                f.write(f"{s_to_tc(start)} --> {s_to_tc(end)}\n")
                f.write(f"{word.word}\n\n")
                lane_idx[lane] = idx + 1

    finally:
        for f in lane_files.values():
            try:
                f.close()
            except Exception:
                pass

    return lane_paths


# -----------------------------
# FCPXML generation (Resolve-importable)
# -----------------------------
def parse_srt_file(srt_path: str) -> List[Tuple[int, int, str]]:
    """
    Returns [(start_ms, end_ms, text), ...]
    """
    def tc_to_ms(tc: str) -> int:
        # "HH:MM:SS,mmm"
        hh, mm, rest = tc.split(":")
        ss, mmm = rest.split(",")
        return (
            int(hh) * 3600 * 1000
            + int(mm) * 60 * 1000
            + int(ss) * 1000
            + int(mmm)
        )

    with open(srt_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    blocks = re.split(r"\n\s*\n", raw)
    out: List[Tuple[int, int, str]] = []

    for b in blocks:
        lines = [ln.rstrip("\n") for ln in b.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        if " --> " not in lines[1]:
            continue

        a, c = lines[1].split(" --> ", 1)
        start_ms = tc_to_ms(a.strip())
        end_ms = tc_to_ms(c.strip())

        text_lines = lines[2:] if len(lines) > 2 else [""]
        text = "\n".join(text_lines).strip()

        out.append((start_ms, end_ms, text))

    return out


def _ms_to_fcpxml_time(ms: int) -> str:
    """
    FCPXML uses rational seconds. We use ms to match SRT timing.
    Examples:
      1500ms -> "3/2s"
      2000ms -> "2s"
    """
    if ms <= 0:
        return "0/1s"
    frac = Fraction(ms, 1000).limit_denominator()
    if frac.denominator == 1:
        return f"{frac.numerator}s"
    return f"{frac.numerator}/{frac.denominator}s"


def write_fcpxml_from_lane_srts(
    lane_srt_paths: List[str],
    out_fcpxml_path: str,
    *,
    fcpxml_version: str,
    width: int,
    height: int,
    fps: int,
    frame_duration_str: Optional[str],
    title_effect_uid: str,
    font: str,
    font_size: str,
    bold: str,
) -> str:
    # Parse all lanes
    lanes: List[List[Tuple[int, int, str]]] = []
    max_end_ms = 0

    for p in lane_srt_paths:
        cues = parse_srt_file(p)
        lanes.append(cues)
        for (_s, e, _t) in cues:
            if e > max_end_ms:
                max_end_ms = e

    total_duration = _ms_to_fcpxml_time(max_end_ms)

    # frameDuration
    if frame_duration_str:
        frame_duration = str(frame_duration_str)
    else:
        frame_duration = f"1/{int(fps)}s"

    fcpxml = ET.Element("fcpxml", {"version": str(fcpxml_version)})

    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        {
            "id": "r0",
            "name": f"FFVideoFormat{height}p{fps}",
            "width": str(int(width)),
            "height": str(int(height)),
            "frameDuration": frame_duration,
        },
    )
    ET.SubElement(
        resources,
        "effect",
        {
            "id": "r1",
            "name": "Basic Title",
            "uid": str(title_effect_uid),
        },
    )

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", {"name": "Lane Import 001"})
    project = ET.SubElement(event, "project", {"name": "Lane Titles 001"})
    sequence = ET.SubElement(
        project,
        "sequence",
        {
            "format": "r0",
            "tcStart": "0/1s",
            "tcFormat": "NDF",
            "duration": total_duration,
        },
    )
    spine = ET.SubElement(sequence, "spine")

    # Base gap to anchor lanes (tracks)
    gap = ET.SubElement(
        spine,
        "gap",
        {
            "name": "Base Gap",
            "offset": "0/1s",
            "start": "0/1s",
            "duration": total_duration,
        },
    )

    style_counter = 0

    for lane_idx, cues in enumerate(lanes, start=1):
        for (start_ms, end_ms, text) in cues:
            dur_ms = max(1, end_ms - start_ms)
            start_t = _ms_to_fcpxml_time(start_ms)
            dur_t = _ms_to_fcpxml_time(dur_ms)

            title = ET.SubElement(
                gap,
                "title",
                {
                    "ref": "r1",
                    "lane": str(lane_idx),
                    "offset": start_t,
                    "start": start_t,
                    "duration": dur_t,
                    "name": "Basic Title",
                    "enabled": "1",
                },
            )

            text_el = ET.SubElement(title, "text", {"roll-up-height": "0"})
            style_id = f"ts{style_counter}"
            style_counter += 1

            ts = ET.SubElement(text_el, "text-style", {"ref": style_id})
            ts.text = text

            tsd = ET.SubElement(title, "text-style-def", {"id": style_id})
            ET.SubElement(
                tsd,
                "text-style",
                {
                    "font": str(font),
                    "fontSize": str(font_size),
                    "bold": str(bold),
                    "italic": "0",
                    "alignment": "center",
                    "lineSpacing": "0",
                },
            )

    # Pretty print (Python 3.9+)
    try:
        ET.indent(fcpxml, space="  ", level=0)
    except Exception:
        pass

    xml_body = ET.tostring(fcpxml, encoding="utf-8").decode("utf-8")
    xml_out = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE fcpxml>\n"
        f"{xml_body}\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_fcpxml_path)), exist_ok=True)
    with open(out_fcpxml_path, "w", encoding="utf-8") as f:
        f.write(xml_out)

    return out_fcpxml_path


# ============================================================
# NEW: DaVinci Resolve Automation (Text+ from SRT lanes)
# ============================================================

def _try_add_resolve_script_paths() -> None:
    """
    Adds common Resolve scripting module paths to sys.path, if not already present.
    """
    candidates = [
        # macOS:
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
        "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/Modules",
        # Windows (common):
        r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
        # Linux (common):
        "/opt/resolve/Developer/Scripting/Modules",
        "/opt/resolve/libs/Fusion/Modules",
    ]
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


def resolve_connect() -> Any:
    """
    Returns resolve app handle or raises a helpful error.
    """
    _try_add_resolve_script_paths()

    try:
        import DaVinciResolveScript as dvr  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import DaVinciResolveScript.\n"
            "Make sure DaVinci Resolve is installed, and the scripting Modules path is on PYTHONPATH.\n"
            "On macOS, try adding:\n"
            "  /Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules\n"
            f"\nUnderlying import error: {e}"
        )

    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError(
            "Failed to get Resolve scripting app handle.\n"
            "Make sure DaVinci Resolve is running (open), then run this script again."
        )
    return resolve


def _call_first(obj: Any, names: List[str], *args: Any, **kwargs: Any) -> Any:
    """
    Try multiple possible method names across Resolve versions.
    """
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"None of these methods exist on {type(obj)}: {names}")


def resolve_get_or_create_project(resolve: Any, project_name: str) -> Any:
    pm = resolve.GetProjectManager()
    if pm is None:
        raise RuntimeError("Resolve.GetProjectManager() returned None.")

    proj = pm.GetCurrentProject()
    if proj and proj.GetName() == project_name:
        return proj

    # Try load existing
    try:
        existing = pm.LoadProject(project_name)
        if existing:
            return existing
    except Exception:
        pass

    # Create new
    created = pm.CreateProject(project_name)
    if created is None:
        # Some builds require switching DB, etc.
        raise RuntimeError(
            f"Could not create project '{project_name}'.\n"
            "Try creating it manually in Resolve first, then rerun."
        )
    return created


def resolve_create_or_get_timeline(project: Any, timeline_name: str) -> Any:
    mp = project.GetMediaPool()
    if mp is None:
        raise RuntimeError("project.GetMediaPool() returned None.")

    # If timeline exists, select it.
    try:
        tls = project.GetTimelineCount()
        for i in range(1, int(tls) + 1):
            tl = project.GetTimelineByIndex(i)
            if tl and tl.GetName() == timeline_name:
                project.SetCurrentTimeline(tl)
                return tl
    except Exception:
        pass

    # Create empty timeline (method name varies)
    tl = None
    try:
        tl = _call_first(mp, ["CreateEmptyTimeline", "CreateTimeline"], timeline_name)
    except Exception:
        # Fallback: some builds use project.CreateTimeline
        try:
            tl = _call_first(project, ["CreateTimeline"], timeline_name)
        except Exception:
            tl = None

    if tl is None:
        raise RuntimeError(
            "Could not create an empty timeline via scripting API.\n"
            "Try switching RESOLVE_MODE to 'import_fcpxml' for the reliable path."
        )

    project.SetCurrentTimeline(tl)
    return tl


def resolve_ensure_video_tracks(timeline: Any, needed_tracks: int) -> None:
    if needed_tracks <= 0:
        return

    # GetTrackCount name varies slightly across versions
    try:
        cur = int(_call_first(timeline, ["GetTrackCount"], "video"))
    except Exception:
        # Some return dict or None; best-effort fallback
        cur = 1

    if cur >= needed_tracks:
        return

    to_add = needed_tracks - cur

    # AddTrack name varies
    for _ in range(to_add):
        try:
            _call_first(timeline, ["AddTrack"], "video")
        except Exception:
            # Some builds use AddTrack("video", index) or AddTrack()
            try:
                _call_first(timeline, ["AddTrack"])
            except Exception as e:
                raise RuntimeError(
                    f"Could not add video tracks to reach {needed_tracks}. "
                    f"Your timeline has {cur}. Error: {e}"
                )


def _fps_to_int(project: Any, fallback: int) -> int:
    # Try to read project setting "timelineFrameRate"
    try:
        ps = project.GetSetting("timelineFrameRate")
        if ps:
            f = float(ps)
            if f > 0:
                return int(round(f))
    except Exception:
        pass
    return int(fallback)


def _ms_to_frames(ms: int, fps: int) -> int:
    # nearest frame
    return int(round((ms / 1000.0) * float(fps)))


def resolve_import_fcpxml(media_pool: Any, fcpxml_path: str) -> Any:
    """
    Most reliable: import the generated FCPXML into Resolve programmatically.
    Returns the newly imported timeline if possible.
    """
    if not os.path.isfile(fcpxml_path):
        raise FileNotFoundError(f"FCPXML not found: {fcpxml_path}")

    # Common method name is ImportTimelineFromFile
    ok = None
    try:
        ok = _call_first(media_pool, ["ImportTimelineFromFile"], fcpxml_path)
    except Exception as e:
        raise RuntimeError(f"MediaPool.ImportTimelineFromFile failed: {e}")

    # Depending on version, ok can be timeline object or bool
    return ok


def _resolve_insert_title_clip(
    project: Any,
    timeline: Any,
    *,
    title_name: str,
    track_index: int,
    record_frame: int,
    duration_frames: int,
) -> Optional[Any]:
    """
    Attempts to insert a Title (preferably Text+) into timeline.
    Returns the inserted TimelineItem if we can find it, else None.

    This is the MOST version-dependent part of Resolve scripting.
    We try multiple approaches.

    Approaches:
    1) MediaPool.AppendToTimeline with generatorName
    2) Timeline.InsertGeneratorIntoTimeline / InsertFusionTitle... (if present)
    """
    mp = project.GetMediaPool()
    if mp is None:
        raise RuntimeError("project.GetMediaPool() returned None.")

    inserted = False

    # --- Approach 1: AppendToTimeline with generatorName (works in some builds) ---
    append_payload_variants = [
        {
            "generatorName": title_name,
            "mediaType": 1,          # 1 video, 2 audio (commonly)
            "trackIndex": track_index,
            "recordFrame": record_frame,
            "duration": duration_frames,
        },
        {
            "generatorName": title_name,
            "trackIndex": track_index,
            "recordFrame": record_frame,
            "duration": duration_frames,
        },
        {
            "generatorName": title_name,
            "recordFrame": record_frame,
            "duration": duration_frames,
        },
    ]

    for payload in append_payload_variants:
        try:
            res = _call_first(mp, ["AppendToTimeline"], [payload])
            if res is not None:
                inserted = True
                break
        except Exception:
            continue

    # --- Approach 2: Timeline insertion functions (playhead-based) ---
    if not inserted:
        # Try to move playhead and insert on active track (some builds ignore trackIndex)
        try:
            # SetCurrentTimecode is common; if absent, ignore
            if hasattr(timeline, "SetCurrentTimecode"):
                # NDF approximation: frame -> tc (hh:mm:ss:ff)
                fps = _fps_to_int(project, int(CONFIG.get("FCPXML_FPS", 30)))
                total = max(0, record_frame)
                hh = total // (3600 * fps)
                total -= hh * 3600 * fps
                mm = total // (60 * fps)
                total -= mm * 60 * fps
                ss = total // fps
                ff = total - ss * fps
                timeline.SetCurrentTimecode(f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}")
        except Exception:
            pass

        # Try insertion methods that exist in some releases
        for method_name in [
            "InsertGeneratorIntoTimeline",
            "InsertFusionGeneratorIntoTimeline",
            "InsertTitleIntoTimeline",
            "InsertFusionTitleIntoTimeline",
        ]:
            fn = getattr(timeline, method_name, None)
            if callable(fn):
                try:
                    # attempt signatures
                    # a) (title_name)
                    # b) (title_name, track_index)
                    # c) (title_name, track_index, duration_frames)
                    try:
                        fn(title_name)
                    except TypeError:
                        try:
                            fn(title_name, track_index)
                        except TypeError:
                            fn(title_name, track_index, duration_frames)
                    inserted = True
                    break
                except Exception:
                    continue

    if not inserted:
        return None

    # Try to find the inserted item by scanning items in track and matching start frame
    try:
        items = _call_first(timeline, ["GetItemListInTrack"], "video", track_index)
        if not items:
            return None
        # Prefer exact start match, else nearest
        best = None
        best_dist = 10**9
        for it in items:
            st = None
            for n in ["GetStart", "GetStartFrame", "GetLeftOffset"]:
                fn = getattr(it, n, None)
                if callable(fn):
                    try:
                        st = int(fn())
                        break
                    except Exception:
                        pass
            if st is None:
                continue
            d = abs(st - record_frame)
            if d < best_dist:
                best_dist = d
                best = it
        return best
    except Exception:
        return None


def _resolve_set_title_text(timeline_item: Any, text: str, tool_candidates: List[str]) -> bool:
    """
    Try to set title text via:
    1) Fusion comp (Text+ / Fusion title): set StyledText on a Text+ tool
    2) Clip property (some non-fusion titles): SetProperty / SetClipProperty variants
    """
    if timeline_item is None:
        return False

    # --- Path A: Fusion comp inside the title ---
    try:
        # comp count can be 0/1+. Methods differ.
        comp = None
        if hasattr(timeline_item, "GetFusionCompByIndex"):
            try:
                comp = timeline_item.GetFusionCompByIndex(1)
            except Exception:
                try:
                    comp = timeline_item.GetFusionCompByIndex(0)
                except Exception:
                    comp = None

        if comp:
            # Try explicit tool names first
            tool = None
            for nm in tool_candidates:
                try:
                    t = comp.FindTool(nm)
                    if t:
                        tool = t
                        break
                except Exception:
                    continue

            # If not found, brute-force: find any tool that has an input named StyledText
            if tool is None:
                try:
                    tl = comp.GetToolList(False)  # returns dict-like in Fusion API
                    if isinstance(tl, dict):
                        for _k, t in tl.items():
                            try:
                                il = t.GetInputList()
                                if isinstance(il, dict) and "StyledText" in il:
                                    tool = t
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass

            if tool is not None:
                # Set StyledText with maximum compatibility
                try:
                    if hasattr(tool, "SetInput"):
                        tool.SetInput("StyledText", text)
                    else:
                        tool["StyledText"] = text
                    return True
                except Exception:
                    pass
    except Exception:
        pass

    # --- Path B: Non-fusion titles might expose properties ---
    for meth in ["SetProperty", "SetClipProperty", "SetProperties"]:
        fn = getattr(timeline_item, meth, None)
        if callable(fn):
            try:
                # common property keys
                for key in ["Text", "StyledText", "Title", "Name"]:
                    try:
                        ok = fn(key, text)
                        if ok is True or ok is None:
                            return True
                    except Exception:
                        continue
            except Exception:
                continue

    return False


def resolve_apply_lane_srts_as_text_plus(
    *,
    lane_srt_paths: List[str],
    project_name: str,
    timeline_name: str,
    title_name: str,
    lane_1_is_top: bool,
    pad_start_frames: int,
    pad_end_frames: int,
    fps_fallback: int,
    fusion_tool_candidates: List[str],
    fcpxml_fallback_path: Optional[str],
) -> None:
    """
    Connect to Resolve, create project/timeline, create video tracks, insert Text+ titles from SRT lanes.
    If insertion fails, optionally fall back to importing FCPXML.
    """
    resolve = resolve_connect()
    project = resolve_get_or_create_project(resolve, project_name)
    media_pool = project.GetMediaPool()
    if media_pool is None:
        raise RuntimeError("project.GetMediaPool() returned None.")

    timeline = resolve_create_or_get_timeline(project, timeline_name)

    fps = _fps_to_int(project, fps_fallback)
    print(f"[resolve] Using FPS={fps} (project timelineFrameRate if available, else fallback)")

    # Parse cues from lanes
    lanes: List[List[Tuple[int, int, str]]] = []
    for p in lane_srt_paths:
        lanes.append(parse_srt_file(p))

    lane_count = len(lanes)
    resolve_ensure_video_tracks(timeline, lane_count)

    # Track mapping
    # If lane_1_is_top=True and we have N tracks:
    #   lane 1 -> track N
    #   lane N -> track 1
    def lane_to_track(lane_idx_1based: int) -> int:
        if not lane_1_is_top:
            return lane_idx_1based
        return (lane_count - lane_idx_1based) + 1

    # Insert clips
    failures = 0
    total = sum(len(c) for c in lanes)
    print(f"[resolve] Inserting {total} title clips ({lane_count} lanes) ...")

    for lane_idx, cues in enumerate(lanes, start=1):
        track_index = lane_to_track(lane_idx)

        for (start_ms, end_ms, text) in tqdm(cues, desc=f"[resolve] Lane {lane_idx}/{lane_count}", leave=True):
            start_fr = _ms_to_frames(start_ms, fps) - int(pad_start_frames)
            end_fr = _ms_to_frames(end_ms, fps) + int(pad_end_frames)

            if end_fr <= start_fr:
                end_fr = start_fr + 1

            dur_fr = max(1, end_fr - start_fr)

            item = _resolve_insert_title_clip(
                project,
                timeline,
                title_name=title_name,
                track_index=int(track_index),
                record_frame=int(max(0, start_fr)),
                duration_frames=int(dur_fr),
            )

            ok_text = _resolve_set_title_text(item, text, fusion_tool_candidates)

            if item is None or not ok_text:
                failures += 1

    if failures == 0:
        print("[resolve] ✅ Done. All Text+ clips inserted and text applied.")
        return

    print(f"[resolve] ⚠️ Finished with {failures} clip(s) where insertion/text-setting failed.")
    if fcpxml_fallback_path:
        print("[resolve] Falling back to importing FCPXML (reliable) ...")
        resolve_import_fcpxml(media_pool, fcpxml_fallback_path)
        print("[resolve] ✅ Imported FCPXML. (This creates stacked title lanes; edit styling in Resolve.)")
    else:
        print("[resolve] No FCPXML fallback path provided; not importing.")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    require_bin("ffmpeg")
    require_bin("ffprobe")

    video_path = os.path.abspath(CONFIG["VIDEO_PATH"])
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"VIDEO_PATH not found: {video_path}")

    out_dir = os.path.abspath(CONFIG["OUT_DIR"])
    prefix = str(CONFIG["OUT_PREFIX"])
    tmp_wav = os.path.abspath(CONFIG["TMP_WAV_PATH"])
    min_cue = float(CONFIG["MIN_CUE_SECONDS"])

    print("[step] Probing video duration...")
    video_end = ffprobe_duration_seconds(video_path)

    print("[step] Extracting audio...")
    extract_audio_wav(video_path, tmp_wav, sr=int(CONFIG["AUDIO_SR"]))

    print("[step] Transcribing (faster-whisper)...")
    words = faster_whisper_words(tmp_wav)

    print("[step] Building batches...")
    batches = make_batches(
        words=words,
        video_end=video_end,
        reset_gap=float(CONFIG["RESET_GAP_SECONDS"]),
        max_words=int(CONFIG["MAX_WORDS_PER_BATCH"]),
        hold_last=float(CONFIG["HOLD_LAST_SECONDS"]),
        cap_hold=float(CONFIG["CAP_HOLD_SECONDS"]),
        split_on_punct=bool(CONFIG["SPLIT_ON_PUNCTUATION"]),
        min_cue=min_cue,
    )

    print("[step] Writing MULTI-LANE SRTs...")
    lane_srt_paths = write_lane_srts(batches, out_dir=out_dir, prefix=prefix, min_cue=min_cue)

    out_fcpxml = ""
    if bool(CONFIG.get("WRITE_FCPXML", True)):
        out_fcpxml = os.path.join(
            out_dir,
            f"{str(CONFIG.get('FCPXML_OUT_NAME', prefix))}.fcpxml"
        )
        print("[step] Writing FCPXML timeline (.fcpxml) for Resolve import...")
        write_fcpxml_from_lane_srts(
            lane_srt_paths,
            out_fcpxml,
            fcpxml_version=str(CONFIG.get("FCPXML_VERSION", "1.8")),
            width=int(CONFIG.get("FCPXML_WIDTH", 1920)),
            height=int(CONFIG.get("FCPXML_HEIGHT", 1080)),
            fps=int(CONFIG.get("FCPXML_FCPXML_FPS", CONFIG.get("FCPXML_FPS", 30))),
            frame_duration_str=CONFIG.get("FCPXML_FRAME_DURATION_STR", None),
            title_effect_uid=str(CONFIG.get("FCPXML_TITLE_EFFECT_UID")),
            font=str(CONFIG.get("FCPXML_FONT", "Helvetica")),
            font_size=str(CONFIG.get("FCPXML_FONT_SIZE", "48")),
            bold=str(CONFIG.get("FCPXML_BOLD", "1")),
        )

    print("\nDone (file generation).")
    print(f"Temp WAV: {tmp_wav}")
    print(f"Words: {len(words)} | Batches: {len(batches)}")

    print("\nGenerated lane SRTs:")
    for p in lane_srt_paths:
        print(" - " + p)

    if out_fcpxml:
        print("\nGenerated FCPXML (import this into Resolve if you want):")
        print(" - " + out_fcpxml)

    # ------------------------------------------------------------
    # NEW: Apply to Resolve (optional)
    # ------------------------------------------------------------
    if bool(CONFIG.get("APPLY_TO_RESOLVE", False)):
        mode = str(CONFIG.get("RESOLVE_MODE", "text_plus_from_srts")).strip().lower()

        if mode == "import_fcpxml":
            if not out_fcpxml:
                raise RuntimeError("RESOLVE_MODE=import_fcpxml but WRITE_FCPXML=False or out_fcpxml not generated.")
            resolve = resolve_connect()
            project = resolve_get_or_create_project(resolve, str(CONFIG["RESOLVE_PROJECT_NAME"]))
            mp = project.GetMediaPool()
            if mp is None:
                raise RuntimeError("project.GetMediaPool() returned None.")
            print("[resolve] Importing FCPXML...")
            resolve_import_fcpxml(mp, out_fcpxml)
            print("[resolve] ✅ Imported FCPXML into Resolve.")
        else:
            print("[resolve] Applying lane SRTs into Resolve as Text+ titles...")
            resolve_apply_lane_srts_as_text_plus(
                lane_srt_paths=lane_srt_paths,
                project_name=str(CONFIG.get("RESOLVE_PROJECT_NAME", "Lane Import Project")),
                timeline_name=str(CONFIG.get("RESOLVE_TIMELINE_NAME", "Lane Titles (Text+)")),
                title_name=str(CONFIG.get("RESOLVE_TITLE_NAME", "Text+")),
                lane_1_is_top=bool(CONFIG.get("RESOLVE_LANE_1_IS_TOP", False)),
                pad_start_frames=int(CONFIG.get("RESOLVE_PAD_START_FRAMES", 0)),
                pad_end_frames=int(CONFIG.get("RESOLVE_PAD_END_FRAMES", 0)),
                fps_fallback=int(CONFIG.get("FCPXML_FPS", 30)),
                fusion_tool_candidates=list(CONFIG.get("RESOLVE_FUSION_TEXT_TOOL_CANDIDATES", ["Text1"])),
                fcpxml_fallback_path=(out_fcpxml or None),
            )

    print("\nResolve usage (manual):")
    print("1) File -> Import -> Timeline -> Import FCPXML... (select the .fcpxml)")
    print("2) Resolve creates a new timeline with stacked title clips (lanes).")
    print("3) Copy/paste those title tracks into your main timeline if you want.")


if __name__ == "__main__":
    main()
