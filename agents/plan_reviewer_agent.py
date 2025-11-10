"""Plan Reviewer Agent - Reviews and optionally updates the editing plan."""

import re
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker
from utils.llm_utils import get_llm_for_planning
from tools.file_tools import is_image_file


def plan_reviewer_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for plan review.
    
    Uses LLM to review the editing plan for completeness, correctness, and clarity.
    Either approves the plan or provides an updated/corrected version.
    """
    artifacts_dir = state.get("artifacts_dir", "artifacts")
    file_manager = FileManager(artifacts_dir=artifacts_dir)
    progress_tracker = ProgressTracker(progress_file=f"{artifacts_dir}/progress.json")
    
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
- Image Files:
  - If CSV has image file in video_path, verify the plan uses the IMAGE clip structure (not video structure):
    * Step 1: Source must clearly indicate it's an image file
    * Step 2: MUST include "Convert image to video clip: duration=X seconds" (duration extracted from additional_operations)
    * Step 3: MUST include "NO Trim" or similar - images have no time range to trim
    * The plan must NOT include a Trim step for image files (unlike video files)
    * All subsequent operations (transitions, effects, overlays, audio) apply to the converted video clip
    * The converted clip duration must match the duration specified in additional_operations
- Trim & Ranges:
  - For VIDEO clips: Each clip includes Trim: start, end.
  - For VIDEO clips: If CSV has end_time=-1, verify the plan explicitly states "end of clip" or "till end" or similar clear indication (not just a numeric value or -1).
  - For VIDEO clips: Any per-feature window (overlay/effect/audio) lies within the trimmed range.
  - For IMAGE clips: There must be NO Trim step (images have no time range to trim).
- Resolution & Aspect Ratio:
  - **CRITICAL CHECK**: Verify the plan explicitly states to NEVER stretch or distort aspect ratio. Verify it instructs to ALWAYS use min(scale_width, scale_height) - never use just one scale factor.
  - Verify the plan explicitly states fixed target as "width=1920 pixels, height=1080 pixels" or "width=1920, height=1080" (not just "1920x1080" or "1920x1080p") to remove ambiguity about which dimension is which.
  - Verify the plan instructs fixed target width=1920, height=1080 (no detection needed).
  - Verify the plan provides EXACT step-by-step instructions:
    (1) Fixed target: width=1920 pixels, height=1080 pixels (explicitly state width and height)
    (2) For each clip: (a) calculate BOTH scale_width=1920/source_width AND scale_height=1080/source_height (explicitly reference target_width=1920 and target_height=1080), (b) ALWAYS use min(scale_width, scale_height) as scale factor (CRITICAL - prevents stretching), (c) scale proportionally (new_width=source_width*scale, new_height=source_height*scale), (d) center (x_offset=(1920-new_width)/2, y_offset=(1080-new_height)/2), (e) REQUIRED: add black bars to reach exactly width=1920, height=1080
    (3) Result: exactly width=1920 pixels, height=1080 pixels, content scaled proportionally (never stretched), centered, with black bars
  - Verify the plan explicitly warns against stretching or using only one scale factor.
  - Ensure the plan makes it clear this normalization happens before concatenation.
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

