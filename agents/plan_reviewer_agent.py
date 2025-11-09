"""Plan Reviewer Agent - Reviews and optionally updates the editing plan."""

import re
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker
from utils.llm_utils import get_llm_for_planning


def plan_reviewer_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for plan review.
    
    Uses LLM to review the editing plan for completeness, correctness, and clarity.
    Either approves the plan or provides an updated/corrected version.
    """
    file_manager = FileManager()
    progress_tracker = ProgressTracker()
    
    progress_tracker.set_stage("PLAN_REVIEW", {})
    
    editing_plan = state.get("editing_plan")
    validated_csv = state.get("validated_csv")
    errors = state.get("errors", [])
    
    if not editing_plan:
        error_msg = "Editing plan not found in state. Run planning agent first."
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        return {
            **state,
            "errors": errors,
            "reviewed_plan": None,
            "progress": progress_tracker.get_progress_dict()
        }
    
    try:
        # Initialize LLM for planning/reviewing
        progress_tracker.add_info("Initializing LLM for plan review...")
        llm = get_llm_for_planning()
        
        # Get plan text
        plan_text = editing_plan.get("plan_text", "")
        total_clips = editing_plan.get("total_clips", 0)
        
        # Prepare full CSV data for review context
        clips_context = ""
        if validated_csv:
            clips_data = validated_csv.get("clips", [])
            total_clips = validated_csv.get("total_clips", len(clips_data))
            
            # Format all clips data similar to planning agent
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
            
            clips_context = f"""
Original CSV specifications (Total clips: {total_clips}):

{''.join(clips_text)}
"""
        
        # Print clips context for debugging
        # print("\n" + "=" * 60)
        # print("PLAN REVIEWER AGENT - Clips Context:")
        # print("=" * 60)
        # print(clips_context)
        # print("=" * 60 + "\n")
        
        # Create review prompt
        system_message = """You are an expert video editing plan reviewer. Review the editing plan against the CSV and enforce the rules below to catch issues early.

Core checks:
- Coverage & Order:
  - All CSV rows are represented in the plan (each CSV row = one clip entry, even if same video file appears multiple times).
  - The same video file can appear multiple times if it's in the CSV multiple times (different clips from same source).
  - Clip indices/order and Source paths match CSV. Final Assembly order is explicit.
  - No extras/omissions: every CSV row has a corresponding clip entry.
- In-place Edits Only:
  - No new/replacement segments within a clip.
  - Transitions/effects/overlays/audio apply in-place to the trimmed clip.
- Exact CSV Values:
  - Use CSV values where specified (times, speeds, volumes, positions, durations, paths).
  - Do not invent assets or parameters unless marked DEFAULT.
- Trim & Ranges:
  - Each clip includes Trim: start, end.
  - Any per-feature window (overlay/effect/audio) lies within the trimmed range.
- Transitions:
  - Per-clip edges: pre/post clear and feasible with needed params (e.g., durations).
  - Between-clip transitions (if any) appear in Final Assembly and don't conflict with per-clip edges.
- Effects:
  - Each effect named with explicit numeric parameters (e.g., brightness=1.1, speed=1.2x, blur=3).
  - No vague terms (e.g., "slightly faster").
  - Effects target the trimmed clip.
- Text Overlays:
  - text, position, size, color, start/end are explicit.
  - Overlay windows are within the trimmed clip.
  - No invented fonts/assets; positioning units are clear.
- Audio Handling:
  - mode is one of keep | replace | mix.
  - External audio path only if present in CSV.
  - Windows allowed: start/end within trimmed clip.
  - replace: default TRIM external audio to the window; LOOP only if CSV requests; silence-fill only if requested.
  - mix: specify external_volume, original_volume, fades (in/out); original remains unless explicitly muted.
  - Audio need not cover entire clip; windows are acceptable.
- Additional Ops:
  - rotate/scale/flip/crop have explicit parameters (degrees, factor or WxH, axis, x,y,w,h).
  - Applied in-place to the trimmed clip.
- Missing/Ambiguous Parameters:
  - Use exactly: MISSING: <param_name> or DEFAULT: <value> (no reasons).
  - Do not silently guess values.
- Final Assembly:
  - Explicit Order: [..].
  - Between-clip transitions (if any) with required params.
  - Export: path=<...>, format=mp4 present and consistent.
- Plan Form & Clarity:
  - No commentary/rationales/test/validation steps in the plan; only plan content.
  - Actionable: all required parameters are concrete (no "approx").
  - No library/API names or implementation details.
- Consistency:
  - Clip references consistent across sections; no contradictions.
  - Units consistent (seconds for time; clear units for positions/sizes).

Output policy:
- If acceptable as-is: respond exactly with
  APPROVED: <short confirmation>
  (Do not include the plan text.)
- If corrections are needed: respond with
  UPDATED:
  <corrected full plan text only>
  (No commentary before/after.)

Be concise and deterministic. Do not add commentary to the plan itself."""

        human_message = f"""Review the following video editing plan against the CSV context. Enforce the rules above and the output policy.

PLAN:
{plan_text}

CSV CONTEXT (Total clips: {total_clips}):
{clips_context}

Return either:
- APPROVED: <short confirmation>
or
- UPDATED:
  <corrected full plan text only>
"""

        # Generate review using LLM
        progress_tracker.add_info("Reviewing plan with LLM...")
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        response = llm.invoke(messages)
        review_response = response.content.strip()
        
        # Parse response to determine if approved or updated
        if review_response.upper().startswith("APPROVED"):
            # Plan is approved, use original plan
            progress_tracker.add_info("Plan approved by reviewer")
            reviewed_plan_text = plan_text
            was_updated = False
        elif review_response.upper().startswith("UPDATED"):
            # Plan was updated, extract the new plan
            progress_tracker.add_info("Plan updated by reviewer")
            # Extract plan text after "UPDATED:" or "UPDATED:\n"
            updated_match = re.search(r'UPDATED:?\s*\n?(.*)', review_response, re.DOTALL | re.IGNORECASE)
            if updated_match:
                reviewed_plan_text = updated_match.group(1).strip()
            else:
                # Fallback: use everything after "UPDATED"
                reviewed_plan_text = re.sub(r'^UPDATED:?\s*', '', review_response, flags=re.IGNORECASE).strip()
            was_updated = True
        else:
            # Ambiguous response, treat as approved but log warning
            progress_tracker.add_info("Warning: Unclear review response, treating as approved")
            reviewed_plan_text = plan_text
            was_updated = False
        
        # Save reviewed plan as text file
        reviewed_plan_text_path = file_manager.save_text("reviewed_plan.txt", reviewed_plan_text)
        progress_tracker.add_info(f"Reviewed plan saved to: {reviewed_plan_text_path}")
        
        # Save reviewed plan as JSON for state management
        reviewed_plan_data = {
            "plan_text": reviewed_plan_text,
            "total_clips": total_clips,
            "was_updated": was_updated,
            "original_plan_path": editing_plan.get("csv_path", "")
        }
        reviewed_plan_json_path = file_manager.save_json("reviewed_plan.json", reviewed_plan_data)
        progress_tracker.add_info(f"Reviewed plan JSON saved to: {reviewed_plan_json_path}")
        
        progress_tracker.set_stage("PLAN_REVIEW", {
            "status": "completed",
            "was_updated": was_updated,
            "total_clips": total_clips
        })
        
        return {
            **state,
            "reviewed_plan": reviewed_plan_data,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
        
    except Exception as e:
        error_msg = f"Error in plan review: {str(e)}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "errors": errors,
            "reviewed_plan": None,
            "progress": progress_tracker.get_progress_dict()
        }

