"""Planning Agent - Generates text-based editing plan from validated CSV."""

import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker
from utils.llm_utils import get_llm_for_planning


def planning_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for planning.
    
    Uses LLM to interpret natural language specifications and create
    a text-based, step-by-step editing plan.
    """
    file_manager = FileManager()
    progress_tracker = ProgressTracker()
    
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
            clip_info = f"""
Clip {clip.get('order', '?')}:
- Source video: {clip.get('video_path', 'N/A')}
- Time range: {clip.get('start_time', 0)}s to {clip.get('end_time', 0)}s
- Transition: {clip.get('transition', 'N/A')}
- Effects: {clip.get('effects', 'N/A')}
- Overlay text: {clip.get('overlay_text', 'None')}
- Overlay description: {clip.get('overlay_description', 'None')}
- Audio description: {clip.get('audio_description', 'N/A')}
- Additional operations: {clip.get('additional_operations', 'None')}
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
- Audio modes: keep | replace | mix (overlay). Audio may target a subrange (window) within the trimmed clip.
  - replace: default to TRIM external audio to the window; LOOP only if CSV requests; silence-fill only if requested.
  - mix: default to TRIM; LOOP only if requested; specify volumes/fades and whether original remains.
  - Audio does not need to cover the full clip; it can apply to a window.
- Missing/ambiguous parameters: mark only with
  - MISSING: <param_name>
  - DEFAULT: <value>
  (no reasons)
- No validations or rationales; output only the plan content.

Preferred structure (you may deviate if needed; no rationale required):
Plan Summary
- Output: <final_output_path>
- Total Clips: <N>

Clip <index> (CSV order)
1) Source: <video_path>
2) Trim: start=<s>, end=<s>
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

Final Assembly
- Order: [<clip1_index>, <clip2_index>, …]
- Between-clip transitions (if any): <type + params>
- Export: path=<final_output_path>, format=mp4"""

        human_message = f"""Create a detailed, actionable plan from the CSV. Use exact CSV values where specified. If something is missing, mark MISSING and propose one DEFAULT (no reasons). Audio may be keep/replace/mix and may use a subrange window.

Total clips: {total_clips}

CSV context:
{clips_context}

Output path: {state.get('output_video_path', 'artifacts/output_video.mp4')}

Requirements:
- In-place operations only (no extra/replacement segments)
- Use CSV values exactly
- Audio: replace/mix may use a window; default to TRIM external audio; LOOP/silence-fill only if requested
- If CSV is incomplete: MISSING and DEFAULT only
- Output only the plan (no commentary); prefer the structure provided, but you may deviate when needed"""

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

