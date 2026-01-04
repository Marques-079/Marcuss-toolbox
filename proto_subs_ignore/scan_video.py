#!/usr/bin/env python3
import os
import sys
import json

# --- 1. RESOLVE SETUP ---
RESOLVE_SCRIPT_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/"
if os.path.exists(RESOLVE_SCRIPT_API):
    sys.path.append(RESOLVE_SCRIPT_API)
    import DaVinciResolveScript as dvr_script

def record_batch_layouts():
    resolve = dvr_script.scriptapp("Resolve")
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline()
    
    if not timeline:
        print("Error: No active timeline found.")
        return

    # This will hold our learned patterns
    # Structure: { "word1_word2_word3": [ {"text": "word1", "x": 0.5, "y": 0.4}, ... ] }
    layout_db = {}
    
    # Load existing memory if it exists to append to it
    memory_path = "layout_memory.json"
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            try:
                layout_db = json.load(f)
            except:
                layout_db = {}

    # 1. Gather all clips from the subtitle tracks (Tracks 2 through 6)
    all_clips = []
    for i in range(2, 7):
        items = timeline.GetItemListInTrack("video", i)
        if items:
            all_clips.extend(items)

    # 2. Group clips by their End Frame (Clips in a batch share the same end time)
    batches = {}
    for clip in all_clips:
        end_time = clip.GetEnd()
        if end_time not in batches:
            batches[end_time] = []
        batches[end_time].append(clip)

    # 3. Process each batch to create a semantic key and record positions
    for end_time, clips in batches.items():
        # Sort clips by start time so the sentence reads correctly
        clips.sort(key=lambda x: x.GetStart())
        
        batch_data = []
        words_found = []
        
        for clip in clips:
            comp = clip.GetFusionCompByIndex(1)
            if not comp: continue
            
            # Find the TextPlus tool
            tool = next((t for t in comp.GetToolList().values() 
                         if t.GetAttrs().get("TOOLS_RegID") == "TextPlus"), None)
            
            if tool:
                txt = tool.GetInput("StyledText").strip().lower()
                center = tool.GetInput("Center") # Returns [X, Y]
                
                # Resolve's GetInput returns center as a dict with keys 1.0 and 2.0
                pos_x = center[1.0]
                pos_y = center[2.0]
                
                words_found.append(txt)
                batch_data.append({
                    "text": txt,
                    "x": pos_x,
                    "y": pos_y
                })
        
        if batch_data:
            # Create a unique key for this specific sequence of words
            sentence_key = "_".join(words_found)
            layout_db[sentence_key] = batch_data
            print(f"Recorded pattern: {sentence_key}")

    # 4. Save to JSON
    with open(memory_path, "w") as f:
        json.dump(layout_db, f, indent=4)
    
    print(f"\nSUCCESS: Saved {len(layout_db)} batch patterns to {memory_path}")

if __name__ == "__main__":
    record_batch_layouts()