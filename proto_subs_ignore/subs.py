#!/usr/bin/env python3
"""
MERGED & OPTIMIZED: 
1. Merges small words (total chars <= 6) into single Text+ items.
2. Adds a 1-frame safety gap to prevent Resolve 'None' track collisions.
3. Verbose logging for API calls.
"""

import os
import sys
import subprocess
import shutil
import math
import time
from dataclasses import dataclass
from typing import List, Optional

# --- 1. SETUP ---
RESOLVE_SCRIPT_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/"
if os.path.exists(RESOLVE_SCRIPT_API):
    sys.path.append(RESOLVE_SCRIPT_API)
    import DaVinciResolveScript as dvr_script
else:
    print("Warning: Could not find DaVinci Resolve Scripting Modules.")

# --- 2. CONFIGURATION ---
CONFIG = {
    "VIDEO_PATH": "/Users/marcus/Downloads/ep2_subs.mov", 
    "TMP_WAV_PATH": "/Users/marcus/Downloads/__tmp_audio_merged.wav",
    
    "TEMPLATE_NAME": "Text+ TEMPLATE", 
    "START_TRACK": 2,                  
    "MAX_TRACKS": 7,  # Increased track pool for safety                 
    
    "FW_MODEL": "small",               
    "FW_DEVICE": "cpu",                
    "FW_COMPUTE_TYPE": "int8",
    "HF_CACHE_DIR": "/Users/marcus/.cache/huggingface",
    
    "RESET_GAP_SECONDS": 0.60,         
    "MAX_WORDS_PER_BATCH": 4,          
    "MIN_CLIP_DURATION_SEC": 0.2, 
    "HOLD_LAST_SECONDS": 0.3,
    "MERGE_CHAR_LIMIT": 6,  # Merge words if total length <= 6
}

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

# --- 4. UTILS ---
def extract_audio_wav(video_path: str, wav_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def run_transcription(wav_path: str) -> List[WordTS]:
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id="Systran/faster-whisper-small", cache_dir=CONFIG["HF_CACHE_DIR"])
    model = WhisperModel(model_path, device=CONFIG["FW_DEVICE"], compute_type=CONFIG["FW_COMPUTE_TYPE"])
    segments, _ = model.transcribe(wav_path, word_timestamps=True, vad_filter=True)
    
    all_words = []
    print("\n--- [WHISPER ENGINE DETECTED] ---")
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            txt = w.word.strip()
            if txt:
                all_words.append(WordTS(word=txt, start=w.start, end=w.end))
                print(f"  [Detect] {w.start:05.2f}s: \"{txt}\"")
    return all_words

def merge_small_words(words: List[WordTS]) -> List[WordTS]:
    """Combines sequential words if combined length <= limit."""
    if not words: return []
    merged = []
    i = 0
    while i < len(words):
        curr = words[i]
        # Peek at next word
        if i + 1 < len(words):
            next_w = words[i+1]
            combined_txt = f"{curr.word} {next_w.word}"
            # Check if merging is viable (length limit + no huge time gap)
            if len(combined_txt) <= CONFIG["MERGE_CHAR_LIMIT"] and (next_w.start - curr.end < 0.3):
                merged.append(WordTS(word=combined_txt, start=curr.start, end=next_w.end))
                i += 2 # Skip next
                continue
        merged.append(curr)
        i += 1
    return merged

def batch_words(words: List[WordTS]) -> List[Batch]:
    if not words: return []
    batches_raw, current = [], []
    for i, w in enumerate(words):
        current.append(w)
        split = False
        if len(current) >= CONFIG["MAX_WORDS_PER_BATCH"]: split = True
        elif w.word[-1] in ".!?": split = True
        elif i + 1 < len(words) and (words[i+1].start - w.end) >= CONFIG["RESET_GAP_SECONDS"]: split = True
        
        if split:
            batches_raw.append(current)
            current = []
    if current: batches_raw.append(current)
    
    final = []
    for i, b_words in enumerate(batches_raw):
        b_start = b_words[0].start
        # SAFETY: Subtract 0.02s (approx 1 frame) to prevent track collision with next batch
        b_end = batches_raw[i+1][0].start - 0.02 if i + 1 < len(batches_raw) else b_words[-1].end + CONFIG["HOLD_LAST_SECONDS"]
        final.append(Batch(start=b_start, end=b_end, words=b_words))
    return final

# --- 6. RESOLVE INTEGRATION ---
def insert_into_resolve(batches: List[Batch]):
    resolve = dvr_script.scriptapp("Resolve")
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline()
    media_pool = project.GetMediaPool()
    
    fps = float(timeline.GetSetting("timelineFrameRate"))
    offset = timeline.GetStartFrame()
    def s_to_f(s): return int(s * fps) + offset

    template = next((c for c in media_pool.GetRootFolder().GetClipList() if CONFIG["TEMPLATE_NAME"] in c.GetName()), None)
    
    clip_ops, meta = [], []
    y_pos = [0.8, 0.6, 0.4, 0.2, 0.1]

    print("\n--- [PREPARING RESOLVE API CALLS] ---")
    for b in batches:
        end_f = s_to_f(b.end)
        for i, w in enumerate(b.words):
            track = CONFIG["START_TRACK"] + i
            start_f = s_to_f(w.start)
            dur = max(int(CONFIG["MIN_CLIP_DURATION_SEC"] * fps), end_f - start_f)
            
            clip_ops.append({
                "mediaPoolItem": template,
                "startFrame": 0, "endFrame": dur,
                "trackIndex": track, "recordFrame": start_f
            })
            meta.append({"txt": w.word, "y": y_pos[i] if i < len(y_pos) else 0.05, "t": track, "s": w.start})
            print(f"  [Queue] Trk {track} | \"{w.word}\"")

    created_clips = media_pool.AppendToTimeline(clip_ops)
    if not created_clips: return

    print("\n--- [WRITING TO FUSION] ---")
    from tqdm import tqdm
    for i, clip in enumerate(tqdm(created_clips)):
        if not clip:
            print(f"  [Skip] Word '{meta[i]['txt']}' at {meta[i]['s']}s FAILED (Collision/Overlap)")
            continue
            
        time.sleep(0.04) # Stability
        comp = clip.GetFusionCompByIndex(1)
        if comp:
            for tool in comp.GetToolList().values():
                if tool.GetAttrs().get("TOOLS_RegID") == "TextPlus":
                    tool.SetInput("StyledText", meta[i]["txt"])
                    tool.SetInput("Center", [0.5, meta[i]["y"]])
                    break
    project.GetProjectManager().SaveProject()

if __name__ == "__main__":
    extract_audio_wav(CONFIG["VIDEO_PATH"], CONFIG["TMP_WAV_PATH"])
    raw_words = run_transcription(CONFIG["TMP_WAV_PATH"])
    merged_words = merge_small_words(raw_words) # <-- NEW MERGE STEP
    batches = batch_words(merged_words)
    insert_into_resolve(batches)
    if os.path.exists(CONFIG["TMP_WAV_PATH"]): os.remove(CONFIG["TMP_WAV_PATH"])
    print("\n--- SUCCESS ---")