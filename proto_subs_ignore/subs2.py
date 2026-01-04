#!/usr/bin/env python3
import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List

# --- 1. SETUP ---
RESOLVE_SCRIPT_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/"
if os.path.exists(RESOLVE_SCRIPT_API):
    sys.path.append(RESOLVE_SCRIPT_API)
    import DaVinciResolveScript as dvr_script

# --- 2. CONFIGURATION ---
CONFIG = {
    "VIDEO_PATH": "/Users/marcus/Downloads/ep2_subs.mov", 
    "TMP_WAV_PATH": "/Users/marcus/Downloads/__tmp_audio_merged.wav",
    "TEMPLATE_NAME": "Text+ TEMPLATE", 
    "START_TRACK": 2,                  
    "FW_DEVICE": "cpu",                
    "FW_COMPUTE_TYPE": "int8",
    "HF_CACHE_DIR": "/Users/marcus/.cache/huggingface",
    
    # Simulation & Emphasis
    "MAX_LINE_CHARS": 14,    
    "MIN_FONT_SIZE": 0.045,   
    "MAX_FONT_SIZE": 0.11,    
    "LINE_1_Y": 0.46,        
    "LINE_2_Y": 0.38,
    "CHAR_WIDTH_FACTOR": 0.31, # Tightened for Open Sans Semibold
    "TRACKING": 1.0,           # Standard spacing
    
    # Priority words (Nouns/Verbs/Emphasis)
    "EMPHASIS_WEIGHT": 1.15,
    "FILLER_WORDS": ["the", "a", "an", "and", "is", "at", "of", "to", "in", "it", "be"],

    "HOLD_LAST_SECONDS": 0.2,
    "RESET_GAP_SECONDS": 0.35, 
}

@dataclass
class WordTS:
    word: str
    start: float
    end: float
    is_emphasis: bool = False

@dataclass
class Batch:
    start: float
    end: float
    words: List[WordTS]

# --- 3. TRANSCRIPTION & SMART SPLITTING ---
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
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            clean_word = w.word.strip()
            if not clean_word: continue
            # Basic Emphasis: If it's long or not a filler word
            emphasis = len(clean_word) > 4 and clean_word.lower() not in CONFIG["FILLER_WORDS"]
            all_words.append(WordTS(word=clean_word, start=w.start, end=w.end, is_emphasis=emphasis))
    return all_words

def batch_words_semantic(words: List[WordTS]) -> List[Batch]:
    """Splits thoughts through the middle and avoids orphan words."""
    batches = []
    curr = []
    curr_len = 0
    
    for i, w in enumerate(words):
        w_len = len(w.word) + 1
        curr.append(w)
        curr_len += w_len
        
        # Look ahead: if the next word ends a sentence, grab it now instead of next batch
        is_end = any(p in w.word for p in ".!?")
        pause = (words[i+1].start - w.end) > CONFIG["RESET_GAP_SECONDS"] if i+1 < len(words) else True
        overflow = curr_len > 26 # 2 lines of ~13 chars
        
        if (is_end or pause or overflow) and len(curr) > 0:
            batches.append(Batch(start=curr[0].start, end=w.end + CONFIG["HOLD_LAST_SECONDS"], words=curr))
            curr, curr_len = [], 0
            
    if curr: batches.append(Batch(start=curr[0].start, end=curr[-1].end, words=curr))
    return batches

# --- 4. THE SENTENCE SIMULATOR ---
def get_unified_layout(batch_words: List[WordTS]):
    """Simulates the sentence as a whole to determine word-perfect offsets."""
    total_text = " ".join([w.word for w in batch_words])
    
    # Split into two lines at logical midpoint
    if len(total_text) <= CONFIG["MAX_LINE_CHARS"]:
        lines = [batch_words]
    else:
        mid = len(total_text) // 2
        split_idx = total_text.rfind(' ', 0, mid + 5)
        l1_txt = total_text[:split_idx].strip()
        
        l1, l2 = [], []
        temp_str = ""
        for w in batch_words:
            if len(temp_str) + len(w.word) <= len(l1_txt) + 1:
                l1.append(w); temp_str += w.word + " "
            else:
                l2.append(w)
        lines = [l1, l2]

    # Calculate Widths & Weighted Scaling
    line_metrics = []
    for line in lines:
        char_count = sum(len(w.word) + 1 for w in line)
        has_emphasis = any(w.is_emphasis for w in line)
        line_metrics.append({"count": char_count, "emphasis": has_emphasis})

    master_count = max(m["count"] for m in line_metrics)
    target_px_width = master_count * CONFIG["MIN_FONT_SIZE"] * CONFIG["CHAR_WIDTH_FACTOR"]

    final_layout = []
    y_coords = [CONFIG["LINE_1_Y"], CONFIG["LINE_2_Y"]]

    for i, line in enumerate(lines):
        m = line_metrics[i]
        # Calculate size to match Master Line width
        base_size = target_px_width / (m["count"] * CONFIG["CHAR_WIDTH_FACTOR"])
        
        # Clamp & Apply Emphasis Weight if this line contains high-value nouns/verbs
        final_size = min(max(base_size, CONFIG["MIN_FONT_SIZE"]), CONFIG["MAX_FONT_SIZE"])
        if m["emphasis"]: 
            final_size = min(final_size * 1.05, CONFIG["MAX_FONT_SIZE"])

        # Sentence-Level Centering
        actual_w = m["count"] * final_size * CONFIG["CHAR_WIDTH_FACTOR"]
        cursor_x = 0.5 - (actual_w / 2)
        
        for w in line:
            # Individual Word Placement based on sentence-string simulation
            w_block_width = (len(w.word) + 1) * final_size * CONFIG["CHAR_WIDTH_FACTOR"]
            final_layout.append({
                "text": w.word, "start": w.start, "size": final_size, "y": y_coords[i],
                "x": cursor_x + (w_block_width / 2)
            })
            cursor_x += w_block_width

    return final_layout

# --- 5. RESOLVE INTEGRATION ---
def insert_into_resolve(batches: List[Batch]):
    resolve = dvr_script.scriptapp("Resolve")
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline()
    media_pool = project.GetMediaPool()
    fps = float(timeline.GetSetting("timelineFrameRate"))
    offset = timeline.GetStartFrame()
    template = next((c for c in media_pool.GetRootFolder().GetClipList() if CONFIG["TEMPLATE_NAME"] in c.GetName()), None)

    all_ops, all_meta = [], []
    for b in batches:
        meta_list = get_unified_layout(b.words)
        end_f = int(b.end * fps) + offset
        
        for i, m in enumerate(meta_list):
            start_f = int(m["start"] * fps) + offset
            dur = max(int(0.12 * fps), end_f - start_f)
            # Strict Track-Locking: Track index correlates to word order
            all_ops.append({
                "mediaPoolItem": template, "trackIndex": CONFIG["START_TRACK"] + i,
                "startFrame": 0, "endFrame": dur, "recordFrame": start_f
            })
            all_meta.append(m)

    clips = media_pool.AppendToTimeline(all_ops)
    for i, clip in enumerate(clips):
        if not clip: continue
        comp = clip.GetFusionCompByIndex(1)
        if comp:
            tool = next((t for t in comp.GetToolList().values() if t.GetAttrs().get("TOOLS_RegID") == "TextPlus"), None)
            if tool:
                tool.SetInput("StyledText", all_meta[i]["text"])
                tool.SetInput("Center", [all_meta[i]["x"], all_meta[i]["y"]])
                tool.SetInput("Size", all_meta[i]["size"])
                # Ensure spacing is neutralized
                tool.SetInput("CharacterSpacing", 1.0)

if __name__ == "__main__":
    extract_audio_wav(CONFIG["VIDEO_PATH"], CONFIG["TMP_WAV_PATH"])
    raw_words = run_transcription(CONFIG["TMP_WAV_PATH"])
    batches = batch_words_semantic(raw_words)
    insert_into_resolve(batches)
    print("--- SUCCESS ---")