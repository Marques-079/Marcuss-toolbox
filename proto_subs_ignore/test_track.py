import sys
import os
import time

# 1. Setup macOS API Path
RESOLVE_SCRIPT_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/"
sys.path.append(RESOLVE_SCRIPT_API)
import DaVinciResolveScript as dvr_script

def comprehensive_multi_track_fix():
    resolve = dvr_script.scriptapp("Resolve")
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    timeline = project.GetCurrentTimeline()
    media_pool = project.GetMediaPool()

    if not timeline:
        print("Error: No active timeline found.")
        return

    # 1. Get current playhead position
    current_tc = timeline.GetCurrentTimecode()
    
    # 2. Find "Text+ TEMPLATE" in the Media Pool
    root_folder = media_pool.GetRootFolder()
    clips = root_folder.GetClipList()
    template_item = None
    
    for clip in clips:
        if "Text+ TEMPLATE" in clip.GetName():
            template_item = clip
            break
            
    if not template_item:
        print("Error: Could not find 'Text+ TEMPLATE' in your Media Pool.")
        return

    # 3. Target tracks 2, 3, 4, 5
    target_tracks = [2, 3, 4, 5]
    print(f"Inserting template at {current_tc} across tracks 2-5...")

    for track_num in target_tracks:
        # Using a list of dictionaries for AppendToTimeline is the strongest way 
        # to bypass the UI's track selection (the red V1 box).
        clip_info = {
            "mediaPoolItem": template_item,
            "trackIndex": int(track_num), # Ensure it's an integer
            "recordFrame": current_tc 
        }
        
        new_clips = media_pool.AppendToTimeline([clip_info])
        
        if new_clips:
            new_clip = new_clips[0]
            # Small delay to allow Resolve to build the Fusion composition
            time.sleep(0.1) 
            
            comp = new_clip.GetFusionCompByIndex(1)
            if comp:
                tools = comp.GetToolList()
                found_text_tool = False
                
                for index in tools:
                    tool = tools[index]
                    attrs = tool.GetAttrs()
                    
                    # SAFELY check for the Tool ID using .get() to avoid KeyErrors
                    # We check both common keys used in Resolve scripting
                    reg_id = attrs.get("TOOLS_RegID") or attrs.get("MAIN_RegName")
                    
                    if reg_id == "TextPlus":
                        # Set the text and the track name
                        tool.SetInput("StyledText", f"Track {track_num}", 0)
                        found_text_tool = True
                        break
                
                if found_text_tool:
                    print(f"Done: Track {track_num}")
                else:
                    print(f"Placed on Track {track_num}, but couldn't find Text+ node inside.")
        else:
            print(f"Failed to place on Track {track_num}. Ensure it isn't locked.")

    project_manager.SaveProject()
    print("Process Complete!")

if __name__ == "__main__":
    comprehensive_multi_track_fix()