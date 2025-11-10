"""Planning Agent - Generates text-based editing plan from validated CSV."""

import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker
from utils.llm_utils import get_llm_for_planning
from tools.file_tools import is_image_file
import re


def planning_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for planning.
    
    Uses LLM to interpret natural language specifications and create
    a text-based, step-by-step editing plan.
    """
    artifacts_dir = state.get("artifacts_dir", "artifacts")
    file_manager = FileManager(artifacts_dir=artifacts_dir)
    progress_tracker = ProgressTracker(progress_file=f"{artifacts_dir}/progress.json")
    
    progress_tracker.set_stage("PLANNING", {})
    
    validated_csv = state.get("validated_csv")
    errors = state.get("errors", [])
    
    if not validated_csv:
        error_msg = "Validated CSV not found in state. Run CSV validation first."
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        return {
            **state,
            "errors": errors,
            "editing_plan": None,
            "progress": progress_tracker.get_progress_dict()
        }
    
    try:
        # Initialize LLM for planning
        progress_tracker.add_info("Initializing LLM for planning...")
        llm = get_llm_for_planning()
        
        # Prepare CSV data for prompt
        clips_data = validated_csv.get("clips", [])
        total_clips = validated_csv.get("total_clips", len(clips_data))
        
        # Format clips data for prompt
        clips_text = []
        for clip in clips_data:
            video_path = clip.get('video_path', 'N/A')
            start_time = clip.get('start_time', 0)
            end_time = clip.get('end_time', 0)
            additional_ops = str(clip.get('additional_operations', 'None'))
            
            # Handle end_time = -1 (till end of clip)
            if end_time == -1:
                time_range = f"{start_time}s to end of clip"
            else:
                time_range = f"{start_time}s to {end_time}s"
            
            # Check if source is an image file
            source_info = f"Source: {video_path}"
            duration_info = ""
            if video_path != 'N/A' and is_image_file(str(video_path)):
                source_info = f"Source: {video_path} (IMAGE)"
                # Try to extract duration from additional_operations
                duration_patterns = [
                    r'for\s+([\d.]+)\s+seconds?',  # "for 0.5 seconds" or "for 0.5 second"
                    r'for\s+([\d.]+)\s*s\b',       # "for 0.5s" or "for 0.5 s"
                    r'duration[:\s]+([\d.]+)\s*seconds?',  # "duration: 0.5 seconds"
                    r'duration[:\s]+([\d.]+)\s*s\b',  # "duration: 0.5s"
                    r'display\s+for\s+([\d.]+)\s+seconds?',  # "display for 0.5 seconds"
                    r'display\s+for\s+([\d.]+)\s*s\b',  # "display for 0.5s"
                    r'relay\s+.*?for\s+([\d.]+)\s*s\b',  # "relay this image for 0.5s"
                    r'([\d.]+)\s+seconds?',  # "0.5 seconds" or "0.5 second"
                    r'([\d.]+)\s*s\b',  # "0.5s" (standalone)
                ]
                for pattern in duration_patterns:
                    match = re.search(pattern, additional_ops, re.IGNORECASE)
                    if match:
                        try:
                            duration_value = float(match.group(1))
                            if duration_value > 0:
                                duration_info = f"\n- Image duration: {duration_value} seconds"
                                break
                        except (ValueError, IndexError):
                            continue
            
            clip_info = f"""
Clip {clip.get('order', '?')}:
- {source_info}{duration_info}
- Time range: {time_range}
- Transition: {clip.get('transition', 'N/A')}
- Effects: {clip.get('effects', 'N/A')}
- Overlay text: {clip.get('overlay_text', 'None')}
- Overlay description: {clip.get('overlay_description', 'None')}
- Audio description: {clip.get('audio_description', 'N/A')}
- Additional operations: {additional_ops}
"""
            clips_text.append(clip_info)
        
        clips_context = "\n".join(clips_text)
        
        # Print clips context for debugging
        # print("\n" + "=" * 60)
        # print("PLANNING AGENT - Clips Context:")
        # print("=" * 60)
        # print(clips_context)
        # print("=" * 60 + "\n")
        
        # Create prompt
        system_message = """You are an expert video editing planner. Create a clear, actionable plan from the CSV that a coding agent can implement 1:1. Do not include commentary, just the plan.

Core rules:
- In-place edits: apply transitions, effects, overlays, and audio changes directly to the trimmed clip; do not create separate replacement segments for the same clip.
- Respect CSV values exactly where specified (times, speeds, volumes, positions, durations). Do not invent assets not present in the CSV.
- Special time handling: If end_time is -1 in the CSV, the plan must specify "Trim: start=<start_time>, end=end of clip" or "Trim: start=<start_time>, end=<clip duration>" to clearly indicate using from start_time till the end of the source video. Do not use -1 in the plan; use explicit "end of clip" language.
- Image files: If video_path points to an image file, the plan structure MUST be different from video clips:
  - Step 1: Source: <image_path> (must clearly indicate this is an image file)
  - Step 2: Convert image to video clip: duration=X seconds (extract duration from additional_operations - look for patterns like "display for X seconds", "for X seconds", "duration: X", "relay for Xs", etc.)
  - Step 3: NO Trim step - images have no time range to trim. The converted video clip will have exactly the duration specified (e.g., 0.5 seconds). Do NOT include a Trim step for image files.
  - Steps 4-8: All subsequent operations (transitions, effects, overlays, audio) apply to the converted video clip. The clip duration is exactly the specified duration (e.g., if duration is 0.5 seconds, the clip is 0.5 seconds long).
  - The plan must make it CRYSTAL CLEAR that: (1) the image file must be converted to a video clip first, (2) the converted clip has the exact duration specified, (3) there is no trim operation for images, (4) all operations apply to the converted video clip.
- Audio modes: keep | replace | mix (overlay). Audio may target a subrange (window) within the trimmed clip.
  - replace: default to TRIM external audio to the window; LOOP only if CSV requests; silence-fill only if requested.
  - mix: default to TRIM; LOOP only if requested; specify volumes/fades and whether original remains.
  - Audio does not need to cover the full clip; it can apply to a window.
- Resolution & Aspect Ratio Normalization: **CRITICAL - Aspect ratio must NEVER be distorted or stretched.** All clips must be normalized to a fixed target resolution: width=1920 pixels, height=1080 pixels (1080p landscape) before concatenation. The plan must clearly instruct the coding agent to implement this EXACT strategy with NO deviations:
  (1) **Fixed Target Resolution**: Target resolution is ALWAYS width=1920 pixels, height=1080 pixels (1080p landscape). No detection needed - this is fixed. Note: width=1920, height=1080 (NOT height=1920, width=1080).
  (2) **Normalize ALL Clips** (videos and images) using these EXACT steps in order - DO NOT skip any step:
    a. **Calculate BOTH scale factors** (you MUST calculate both, never use just one):
       - scale_width = target_width / source_width where target_width = 1920
       - scale_height = target_height / source_height where target_height = 1080
    b. **ALWAYS use the SMALLER scale factor** (this is CRITICAL to prevent stretching):
       - scale = min(scale_width, scale_height)
       - **NEVER use scale_width alone or scale_height alone** - this will cause stretching
       - **NEVER stretch to fill the frame** - always use the smaller factor
    c. **Scale the clip proportionally**:
       - new_width = source_width * scale
       - new_height = source_height * scale
       - This preserves aspect ratio (no distortion)
    d. **Center the scaled clip** (the scaled clip will be smaller than or equal to width=1920, height=1080):
       - x_offset = (target_width - new_width) / 2 where target_width = 1920
       - y_offset = (target_height - new_height) / 2 where target_height = 1080
    e. **REQUIRED: Add black bars to fill remaining area**:
       - The scaled clip will NOT fill the entire frame (unless aspect ratios match exactly)
       - You MUST add black bars (letterbox top/bottom OR pillarbox left/right) to reach exactly width=1920, height=1080
       - Final clip MUST be exactly width=1920 pixels, height=1080 pixels with scaled content centered and black bars filling the rest
  (3) **Examples**:
    - Portrait clip (width=1080, height=1920): scale_width=1920/1080=1.778, scale_height=1080/1920=0.5625 → use 0.5625 (smaller) → scaled to width=607.5, height=1080 → center → add black bars left/right → final: width=1920, height=1080
    - Landscape clip (width=1280, height=720): scale_width=1920/1280=1.5, scale_height=1080/720=1.5 → use 1.5 → scaled to width=1920, height=1080 → no padding needed → final: width=1920, height=1080
    - Large 4K clip (width=3840, height=2160): scale_width=1920/3840=0.5, scale_height=1080/2160=0.5 → use 0.5 → scaled to width=1920, height=1080 → no padding needed → final: width=1920, height=1080
  (4) **Result**: All clips have EXACTLY width=1920 pixels, height=1080 pixels. Content is scaled proportionally (never stretched), centered, with black bars filling remaining area. The plan must make this strategy CRYSTAL CLEAR with explicit warnings against stretching.
- Missing/ambiguous parameters: mark only with
  - MISSING: <param_name>
  - DEFAULT: <value>
  (no reasons)
- No validations or rationales; output only the plan content.

Preferred structure (you may deviate if needed; no rationale required):
Plan Summary
- Output: <final_output_path>
- Total Clips: <N>
- Resolution Normalization: **CRITICAL - Preserve aspect ratio, NEVER stretch.** Normalize all clips to fixed target: width=1920 pixels, height=1080 pixels. (1) Calculate scale_width=1920/source_width and scale_height=1080/source_height, (2) ALWAYS use min(scale_width, scale_height) as scale factor, (3) Scale proportionally, (4) Center scaled clip, (5) Add black bars to reach exactly width=1920, height=1080

For VIDEO clips:
Clip <index> (CSV order)
1) Source: <video_path>
2) Trim: start=<s>, end=<s> (or "end of clip" if end_time=-1)
3) Transitions (in-place, edges): pre=<type or none>(params), post=<type or none>(params)
4) Effects (or "none"): name=params (e.g., brightness=1.1, speed=1.2x, blur=3)
5) Text Overlays (or "none"): text="…", position=…, fontsize=…, color=…, start=<s>, end=<s>
6) Audio:
   - mode=<keep|replace|mix>
   - if external: audio_path=<path>
   - window: start=<s>, end=<s> (optional)
   - external_handling=<trim | loop_when_requested | silence_fill_when_requested>
   - mix: external_volume=<>, original_volume=<>, fades: in=<s>, out=<s>
7) Additional Ops (or "none"): rotate=<deg>, scale=<factor or WxH>, flip=<…>, crop=<x,y,w,h>
8) Save: in-place

For IMAGE clips (different structure - NO trim step):
Clip <index> (CSV order)
1) Source: <image_path> (image file)
2) Convert image to video clip: duration=<X> seconds (extracted from additional_operations)
3) NO Trim - converted clip duration is exactly <X> seconds
4) Transitions (in-place, edges): pre=<type or none>(params), post=<type or none>(params) (apply to converted clip)
5) Effects (or "none"): name=params (apply to converted clip)
6) Text Overlays (or "none"): text="…", position=…, fontsize=…, color=…, start=<s>, end=<s> (times relative to converted clip duration)
7) Audio:
   - mode=<keep|replace|mix>
   - if external: audio_path=<path>
   - window: start=<s>, end=<s> (relative to converted clip duration)
   - external_handling=<trim | loop_when_requested | silence_fill_when_requested>
   - mix: external_volume=<>, original_volume=<>, fades: in=<s>, out=<s>
8) Additional Ops (or "none"): rotate=<deg>, scale=<factor or WxH>, flip=<…>, crop=<x,y,w,h> (apply to converted clip)
9) Save: in-place

Final Assembly
- Order: [<clip1_index>, <clip2_index>, …]
- Between-clip transitions (if any): <type + params>
- Export: path=<final_output_path>, format=mp4"""

        human_message = f"""Create a detailed, actionable plan from the CSV. Use exact CSV values where specified. If something is missing, mark MISSING and propose one DEFAULT (no reasons). Audio may be keep/replace/mix and may use a subrange window.

Total clips: {total_clips}

CSV context:
{clips_context}

Output path: {state.get('output_video_path', f'{artifacts_dir}/output_video.mp4')}

Requirements:
- In-place operations only (no extra/replacement segments)
- Use CSV values exactly
- If end_time is -1: specify "end of clip" or "till end" explicitly in the plan (not -1)
- If source is an image: 
  * MUST use the IMAGE clip structure (not video structure)
  * MUST include "Convert image to video clip: duration=X seconds" as step 2
  * MUST include "NO Trim" as step 3 (images have no time range to trim)
  * The converted clip duration is exactly the specified duration
  * All subsequent operations apply to the converted video clip
- Audio: replace/mix may use a window; default to TRIM external audio; LOOP/silence-fill only if requested
- If CSV is incomplete: MISSING and DEFAULT only
- Output only the plan (no commentary); use the appropriate structure (VIDEO or IMAGE) for each clip"""

        # Generate plan using LLM
        progress_tracker.add_info("Generating editing plan with LLM...")
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        response = llm.invoke(messages)
        plan_text = response.content
        
        # Save plan as text file
        plan_text_path = file_manager.save_text("editing_plan.txt", plan_text)
        progress_tracker.add_info(f"Plan saved to: {plan_text_path}")
        
        # Save plan as JSON for state management
        plan_data = {
            "plan_text": plan_text,
            "total_clips": total_clips,
            "csv_path": validated_csv.get("csv_path", "")
        }
        plan_json_path = file_manager.save_json("editing_plan.json", plan_data)
        progress_tracker.add_info(f"Plan JSON saved to: {plan_json_path}")
        
        progress_tracker.set_stage("PLANNING", {
            "status": "completed",
            "total_clips": total_clips,
            "plan_text_path": plan_text_path
        })
        
        return {
            **state,
            "editing_plan": plan_data,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
        
    except Exception as e:
        error_msg = f"Error in planning: {str(e)}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "errors": errors,
            "editing_plan": None,
            "progress": progress_tracker.get_progress_dict()
        }

