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
    
    # Strict Typography (FORCED SIZE)
    "FONT_FACE": "Helvetica",
    "FONT_STYLE": "Bold",
    "FIXED_FONT_SIZE": 0.0354,  # Replaced MIN/MAX with fixed value
    "LETTER_WIDTH_FACTOR": 0.40, 
    "WORD_GAP_FACTOR": 0.50,     
    
    # Positioning & Logic
    "MAX_LINE_CHARS": 14,        
    "LINE_1_Y": 0.46,        
    "LINE_2_Y": 0.38,
    "HOLD_LAST_SECONDS": 0.3,
    "RESET_GAP_SECONDS": 0.4, 
    
    "FILLER_WORDS": {"it's", "it", "not", "the", "a", "an", "is", "am", "are", "of", "to", "in", "that's", "that", "and", "which", "for", "with", "on"},
}

@dataclass
class WordTS:
    word: str
    start: float
    end: float
    is_important: bool

@dataclass
class Batch:
    start: float
    end: float
    words: List[WordTS]

# --- 3. TRANSCRIPTION & SEMANTIC PROCESSING ---
def extract_audio_wav(video_path: str, wav_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def run_transcription(wav_path: str) -> List[WordTS]:
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id="Systran/faster-whisper-small", cache_dir=CONFIG["HF_CACHE_DIR"])
    model = WhisperModel(model_path, device=CONFIG["FW_DEVICE"], compute_type=CONFIG["FW_COMPUTE_TYPE"])
    segments, _ = model.transcribe(wav_path, word_timestamps=True, vad_filter=True)
    
    raw_list = []
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            raw_list.append({"txt": w.word.strip().lower(), "s": w.start, "e": w.end})

    processed = []
    i = 0
    while i < len(raw_list):
        txt = raw_list[i]["txt"]
        if txt == "pre" and i+1 < len(raw_list) and "processing" in raw_list[i+1]["txt"]:
            txt = "pre-processing"
            end_t = raw_list[i+1]["e"]
            i += 1
        else:
            end_t = raw_list[i]["e"]
            
        important = txt not in CONFIG["FILLER_WORDS"]
        processed.append(WordTS(word=txt, start=raw_list[i]["s"], end=end_t, is_important=important))
        i += 1
    return processed

def batch_words_semantic(words: List[WordTS]) -> List[Batch]:
    batches, curr, curr_len = [], [], 0
    for i, w in enumerate(words):
        if curr_len + len(w.word) > (CONFIG["MAX_LINE_CHARS"] * 1.8):
            batches.append(Batch(start=curr[0].start, end=curr[-1].end + CONFIG["HOLD_LAST_SECONDS"], words=curr))
            curr, curr_len = [], 0
        
        curr.append(w)
        curr_len += len(w.word) + 1
        pause = (words[i+1].start - w.end) > CONFIG["RESET_GAP_SECONDS"] if i+1 < len(words) else True
        if pause:
            batches.append(Batch(start=curr[0].start, end=w.end + CONFIG["HOLD_LAST_SECONDS"], words=curr))
            curr, curr_len = [], 0
    if curr: batches.append(Batch(start=curr[0].start, end=curr[-1].end, words=curr))
    return batches

# --- 4. THE LAYOUT ENGINE ---
def get_simulated_layout(batch_words: List[WordTS]):
    lines = []
    current_line = []
    current_chars = 0
    for w in batch_words:
        if current_chars + len(w.word) > CONFIG["MAX_LINE_CHARS"] and current_line:
            lines.append(current_line)
            current_line = [w]
            current_chars = len(w.word)
        else:
            current_line.append(w)
            current_chars += len(w.word) + 1
    if current_line: lines.append(current_line)

    layout = []
    y_coords = [CONFIG["LINE_1_Y"], CONFIG["LINE_2_Y"]]
    line_size = CONFIG["FIXED_FONT_SIZE"]
    
    for i, line in enumerate(lines[:2]):
        char_units = sum(len(w.word) for w in line)
        gap_units = (len(line) - 1) * CONFIG["WORD_GAP_FACTOR"] if len(line) > 1 else 0
        
        total_w = (char_units * line_size * CONFIG["LETTER_WIDTH_FACTOR"]) + (gap_units * line_size)
        cursor_x = 0.5 - (total_w / 2)
        
        for w in line:
            word_w = len(w.word) * line_size * CONFIG["LETTER_WIDTH_FACTOR"]
            layout.append({
                "text": w.word, 
                "size": line_size, 
                "y": y_coords[i],
                "x": cursor_x + (word_w / 2)
            })
            cursor_x += (word_w + (CONFIG["WORD_GAP_FACTOR"] * line_size))
            
    return layout

# --- 5. RESOLVE INTEGRATION (UPDATED FOR CASCADE) ---
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
        # Get the layout positions
        meta_list = get_simulated_layout(b.words)
        
        # The unified end point for the entire thought/batch
        batch_end_f = int(b.end * fps) + offset
        
        for i, m in enumerate(meta_list):
            # RESTORE CASCADE: Use the individual word's start time
            # We match the word in meta_list to the original word in b.words for timing
            word_data = b.words[i] if i < len(b.words) else b.words[-1]
            start_f = int(word_data.start * fps) + offset
            
            # Duration extends from the word's start to the end of the batch
            dur = max(2, batch_end_f - start_f)
            
            all_ops.append({
                "mediaPoolItem": template, 
                "trackIndex": CONFIG["START_TRACK"] + i, # Staggered tracks
                "startFrame": 0, 
                "endFrame": dur, 
                "recordFrame": start_f
            })
            all_meta.append(m)

    clips = media_pool.AppendToTimeline(all_ops)
    # ... (rest of the Fusion tool settings remain the same)
    for i, clip in enumerate(clips):
        if not clip: continue
        comp = clip.GetFusionCompByIndex(1)
        if comp:
            tool = next((t for t in comp.GetToolList().values() if t.GetAttrs().get("TOOLS_RegID") == "TextPlus"), None)
            if tool:
                tool.SetInput("Font", CONFIG["FONT_FACE"])
                tool.SetInput("Style", CONFIG["FONT_STYLE"])
                tool.SetInput("StyledText", all_meta[i]["text"])
                tool.SetInput("Center", [all_meta[i]["x"], all_meta[i]["y"]])
                tool.SetInput("Size", all_meta[i]["size"])
                
                # SHADOW ELEMENT
                tool.SetInput("SelectElement", 2)
                tool.SetInput("Enabled2", 1)
                tool.SetInput("Opacity2", 0.732)
                tool.SetInput("Thickness2", 0.4)
                tool.SetInput("Red2", 0.0); tool.SetInput("Green2", 0.0); tool.SetInput("Blue2", 0.0)
                tool.SetInput("SoftnessX2", 9.45)
                tool.SetInput("SoftnessY2", 4.25)

if __name__ == "__main__":
    extract_audio_wav(CONFIG["VIDEO_PATH"], CONFIG["TMP_WAV_PATH"])
    raw_words = run_transcription(CONFIG["TMP_WAV_PATH"])
    batches = batch_words_semantic(raw_words)
    insert_into_resolve(batches)
    if os.path.exists(CONFIG["TMP_WAV_PATH"]): os.remove(CONFIG["TMP_WAV_PATH"])
    print(f"--- SUCCESS: FONT FIXED AT {CONFIG['FIXED_FONT_SIZE']} ---")