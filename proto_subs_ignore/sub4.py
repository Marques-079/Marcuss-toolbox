#!/usr/bin/env python3
import os
import sys
import subprocess
import time
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
    "START_TRACK": 3,                  
    "FW_DEVICE": "cpu",                
    "FW_COMPUTE_TYPE": "int8",
    "HF_CACHE_DIR": "/Users/marcus/.cache/huggingface",
    
    # Typography
    "FONT_FACE": "Helvetica",
    "FONT_STYLE": "Bold",
    "FIXED_FONT_SIZE": 0.354,       # Standardizing on your requested size
    "LETTER_WIDTH_FACTOR": 0.40, 
    "WORD_GAP_FACTOR": 0.50,     
    
    # Layout & Semantic Logic
    "MAX_LINE_CHARS": 15,          
    "MAX_BATCH_LINES": 2,          
    "LINE_1_Y": 0.46,        
    "LINE_2_Y": 0.38,
    "HOLD_LAST_SECONDS": 0.3,
    "MIN_WORDS_PER_BATCH": 3,
    
    "BREAK_WORDS": {"and", "but", "or", "so", "because", "then", "if", "when", "however"},
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

# --- 3. PROCESSING ---
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
            raw_list.append(WordTS(word=w.word.strip(), start=w.start, end=w.end))
    return raw_list

def batch_words_by_idea(words: List[WordTS]) -> List[Batch]:
    batches = []
    curr = []
    
    for i, w in enumerate(words):
        curr.append(w)
        curr_chars = sum(len(x.word) + 1 for x in curr)
        
        should_break = False
        # Break if we hit char limit
        if curr_chars >= (CONFIG["MAX_LINE_CHARS"] * CONFIG["MAX_BATCH_LINES"]):
            should_break = True
        
        # Break for semantic reasons (The "Idea" capture)
        elif len(curr) >= CONFIG["MIN_WORDS_PER_BATCH"]:
            if i + 1 < len(words):
                nxt = words[i+1].word.lower()
                pause = words[i+1].start - w.end
                # Break at conjunctions or significant pauses
                if nxt in CONFIG["BREAK_WORDS"] or pause > 0.45:
                    should_break = True
            else:
                should_break = True

        if should_break:
            batches.append(Batch(start=curr[0].start, end=curr[-1].end + CONFIG["HOLD_LAST_SECONDS"], words=curr))
            curr = []
            
    if curr:
        batches.append(Batch(start=curr[0].start, end=curr[-1].end, words=curr))
    return batches

# --- 4. LAYOUT ENGINE ---
def get_layout(batch_words: List[WordTS]):
    lines, curr_line, curr_chars = [], [], 0
    
    for w in batch_words:
        if curr_chars + len(w.word) > CONFIG["MAX_LINE_CHARS"] and curr_line:
            lines.append(curr_line)
            curr_line, curr_chars = [w], len(w.word)
        else:
            curr_line.append(w)
            curr_chars += len(w.word) + 1
    if curr_line: lines.append(curr_line)

    layout = []
    y_coords = [CONFIG["LINE_1_Y"], CONFIG["LINE_2_Y"]]
    sz = CONFIG["FIXED_FONT_SIZE"]
    
    for i, line in enumerate(lines[:2]): 
        total_w = (sum(len(w.word) for w in line) * sz * CONFIG["LETTER_WIDTH_FACTOR"]) + \
                  ((len(line)-1) * CONFIG["WORD_GAP_FACTOR"] * sz)
        cursor_x = 0.5 - (total_w / 2)
        
        for w in line:
            w_width = len(w.word) * sz * CONFIG["LETTER_WIDTH_FACTOR"]
            layout.append({
                "text": w.word, "start": w.start, 
                "x": cursor_x + (w_width / 2), "y": y_coords[i]
            })
            cursor_x += (w_width + (CONFIG["WORD_GAP_FACTOR"] * sz))
    return layout

# --- 5. RESOLVE INTEGRATION ---
def insert_into_resolve(batches: List[Batch]):
    resolve = dvr_script.scriptapp("Resolve")
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline()
    media_pool = project.GetMediaPool()
    fps = float(timeline.GetSetting("timelineFrameRate"))
    offset = timeline.GetStartFrame()
    template = next((c for c in media_pool.GetRootFolder().GetClipList() if CONFIG["TEMPLATE_NAME"] in c.GetName()), None)

    if not template:
        print(f"Error: Template '{CONFIG['TEMPLATE_NAME']}' not found in Root Folder.")
        return

    for b in batches:
        meta_list = get_layout(b.words)
        batch_end_f = int(b.end * fps) + offset
        
        for i, m in enumerate(meta_list):
            start_f = int(m["start"] * fps) + offset
            dur = max(3, batch_end_f - start_f)
            
            # Spread words across tracks to avoid collision/overwriting
            # Track logic: StartTrack + (0 or 1 for Line) + i*2 for depth
            track = CONFIG["START_TRACK"] + (1 if m["y"] == CONFIG["LINE_2_Y"] else 0) + (i * 2)
            
            clip_info = [{
                "mediaPoolItem": template, "trackIndex": track,
                "startFrame": 0, "endFrame": dur, "recordFrame": start_f
            }]
            
            clips = media_pool.AppendToTimeline(clip_info)
            if not clips: continue
            
            clip = clips[0]
            # RETRY LOGIC: Fixes the 'NoneType' GetFusionCompByIndex error
            comp = None
            for attempt in range(5):
                comp = clip.GetFusionCompByIndex(1)
                if comp: break
                time.sleep(0.05)
            
            if comp:
                tools = comp.GetToolList().values()
                tool = next((t for t in tools if t.GetAttrs().get("TOOLS_RegID") == "TextPlus"), None)
                if tool:
                    comp.Lock()
                    tool.SetInput("StyledText", m["text"])
                    tool.SetInput("Center", [m["x"], m["y"]])
                    tool.SetInput("Size", CONFIG["FIXED_FONT_SIZE"])
                    tool.SetInput("Font", CONFIG["FONT_FACE"])
                    tool.SetInput("Style", CONFIG["FONT_STYLE"])
                    
                    # Optional: Add a subtle black outline for readability
                    tool.SetInput("SelectElement", 2)
                    tool.SetInput("Enabled2", 1)
                    tool.SetInput("Thickness2", 0.1)
                    tool.SetInput("Red2", 0.0); tool.SetInput("Green2", 0.0); tool.SetInput("Blue2", 0.0)
                    comp.Unlock()

if __name__ == "__main__":
    extract_audio_wav(CONFIG["VIDEO_PATH"], CONFIG["TMP_WAV_PATH"])
    raw_words = run_transcription(CONFIG["TMP_WAV_PATH"])
    semantic_batches = batch_words_by_idea(raw_words)
    insert_into_resolve(semantic_batches)
    if os.path.exists(CONFIG["TMP_WAV_PATH"]): os.remove(CONFIG["TMP_WAV_PATH"])
    print("--- SUCCESS: SEMANTIC IDEAS SYNCED ---")