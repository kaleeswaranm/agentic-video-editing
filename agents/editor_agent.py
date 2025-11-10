"""Editor Agent - Generates Python code for video editing based on reviewed plan."""

from langchain_core.messages import SystemMessage, HumanMessage

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker
from utils.llm_utils import get_llm_for_coding


def editor_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for code generation.
    
    Uses LLM (GPT-5 for coding) to generate Python code that implements
    the video editing workflow from the reviewed plan.
    """
    artifacts_dir = state.get("artifacts_dir", "artifacts")
    file_manager = FileManager(artifacts_dir=artifacts_dir)
    progress_tracker = ProgressTracker(progress_file=f"{artifacts_dir}/progress.json")
    
    progress_tracker.set_stage("CODE_GENERATION", {})
    
    reviewed_plan = state.get("reviewed_plan")
    output_video_path = state.get("output_video_path", f"{artifacts_dir}/output_video.mp4")
    errors = state.get("errors", [])
    
    if not reviewed_plan:
        error_msg = "Reviewed plan not found in state. Run plan reviewer first."
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        return {
            **state,
            "errors": errors,
            "generated_code": None,
            "progress": progress_tracker.get_progress_dict()
        }
    
    try:
        # Initialize LLM for coding
        progress_tracker.add_info("Initializing LLM for code generation...")
        llm = get_llm_for_coding()
        
        # Get plan text
        plan_text = reviewed_plan.get("plan_text", "")
        
        # Create code generation prompt
        system_message = """You are an expert Python video editing developer. Your task is to generate complete, executable Python code that implements a video editing workflow.

Requirements:
1. Use MoviePy library for video editing operations
   **IMPORTANT: Use MoviePy version 1.0.3 API. Use the standard MoviePy 1.0.3 API patterns.**
2. **Available libraries in the environment (use these as much as possible):**
   - MoviePy 1.0.3 (from moviepy.editor import ...) - Primary video editing library
   - NumPy (import numpy) - Array operations and numerical computations
   - Pillow/PIL (from PIL import Image) - Image processing
   - pillow-heif (from pillow_heif import register_heif_opener) - For HEIC/HEIF image format support. Use: register_heif_opener() before using PIL.Image.open() for HEIC/HEIF files
   - imageio (import imageio) - Video file I/O operations
   - imageio-ffmpeg - FFmpeg integration (used by imageio)
   - scipy (import scipy) - Scientific computing for advanced operations
   - pydub (from pydub import AudioSegment) - Audio processing (if needed for advanced audio operations)
   - opencv-python (import cv2) - Advanced image/video processing, effects, and transformations
   - tqdm (from tqdm import tqdm) - Progress bars for long operations
   - pandas (import pandas) - Data manipulation if needed
   Prefer using these standard libraries over custom implementations.
3. Generate complete, runnable Python script
4. Follow the plan step-by-step
5. Include all necessary imports (all libraries above are already installed in the environment)
6. Handle file paths correctly (use pathlib or os.path)
7. Create intermediate directories if needed (e.g., <artifacts_dir>/temp/)
8. **DO NOT use try/except blocks** - Let errors propagate naturally so they can be detected and fixed
9. Add progress logging using print statements
10. Make code readable with comments
11. Process each clip according to the plan
12. Apply all effects, transitions, overlays, and audio operations
13. Concatenate clips in order
14. Export final video to the specified output path

Code structure:
- Import necessary libraries (MoviePy 1.0.3 is already installed)
- Define main execution function or script
- Process each clip sequentially
- Handle all operations: trim, effects, transitions, overlays, audio
- Concatenate and export final video
- **Do not wrap code in try/except blocks** - errors should propagate so they can be detected

Important:
- Use MoviePy version 1.0.3 API patterns
- Use standard MoviePy 1.0.3 imports (from moviepy.editor import ...)
- **Prefer using available libraries**: MoviePy, NumPy, Pillow, pillow-heif, imageio, scipy, opencv-python, pydub, tqdm
- **For HEIC/HEIF images**: Use pillow-heif library - (1) Import: from pillow_heif import register_heif_opener, (2) Register: call register_heif_opener() before opening images, (3) Then use PIL.Image.open() to load HEIC/HEIF images normally. Do NOT use opencv (cv2) for HEIC images - use pillow-heif instead.
- **RESOLUTION NORMALIZATION (MANDATORY)**:
  - Target resolution is fixed: width=1920, height=1080 (1080p landscape).
  - Compute scale factors:
    * scale_width = 1920 / source_width
    * scale_height = 1080 / source_height
    * scale = min(scale_width, scale_height)  # CRITICAL: prevents stretching
  - Define and USE a helper exactly like this (or exact equivalent):
    def normalize_resolution(clip, target_w=1920, target_h=1080):
        src_w, src_h = clip.size
        scale = min(target_w / src_w, target_h / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        resized = clip.resize((new_w, new_h))
        bg = ColorClip(size=(target_w, target_h), color=(0, 0, 0), duration=resized.duration)
        out = CompositeVideoClip([bg, resized.set_position(("center","center"))], size=(target_w, target_h)).set_duration(resized.duration)
        print(f"[normalize] src=({src_w},{src_h}) scaled=({new_w},{new_h}) final={out.size}")
        assert out.size == (target_w, target_h)
        return out
  - Resize proportionally when computing new size:
    * new_w = int(source_width * scale), new_h = int(source_height * scale)
  - Center + pad on black background (MoviePy 1.0.3-safe) by calling normalize_resolution; the bg ColorClip MUST be included in CompositeVideoClip layers.
  - After normalization: print pre/scale/final sizes and assert clip.size == (1920, 1080).
  - Call normalize_resolution:
    * immediately after trim/convert (for every clip)
    * again after any overlays/effects that wrap in CompositeVideoClip (e.g., text); re-assert final size
  - DO NOT:
    * call clip.resize((1920,1080)) directly (would distort aspect ratio)
    * use set_make_frame for normalization; prefer CompositeVideoClip + ColorClip (or on_color with assert)
- **COORDINATE SYSTEM (CRITICAL)**:
  - MoviePy sizes are (width, height); NumPy frames are (height, width, channels).
  - If composing manually with NumPy arrays: bg[y:y+new_h, x:x+new_w, :] = frame (rows=y, cols=x).
- **ORIENTATION HANDLING (MANDATORY, EXIFTOOL ONLY)**:
  - Detect rotation using EXIFTool only, via subprocess:
    * cmd: exiftool -Composite:Rotation -s3 <path>
    * parse numeric degrees; if missing, use 0
  - Do NOT rotate pixels for orientation correction. Use rotation only to determine scaling geometry:
    * If rotation ∈ {90, 270}: effective scale dimensions = (eff_w, eff_h) = (source_height, source_width)
    * Else: (eff_w, eff_h) = (source_width, source_height)
    * scale = min(1920/eff_w, 1080/eff_h)
    * Compute resized size WITHOUT rotating pixels:
      - If rotation ∈ {90, 270}: (new_w, new_h) = (int(source_height*scale), int(source_width*scale))  # e.g., 607x1080
      - Else: (new_w, new_h) = (int(source_width*scale), int(source_height*scale))
  - Center on ColorClip background to width=1920, height=1080; print src/eff/scale/resized/final sizes and assert final size == (1920,1080).
  - On export, set ffmpeg_params=["-metadata:s:v:0","rotate=0"] to avoid writing rotation metadata.
  - DO NOT use clip.rotate(...) for orientation correction.
- Handle audio operations (volume, fade, replacement) using MoviePy or pydub if needed
- Apply video effects (brightness, contrast, blur, speed, color correction) using MoviePy, opencv-python, or scipy
- Add text overlays with proper positioning and styling using MoviePy
- Handle transitions between clips using MoviePy
- Use tqdm for progress bars on long operations
- Clean up intermediate files if needed
- **No try/except blocks** - let Python raise exceptions naturally so errors can be caught and fixed"""

        human_message = f"""Generate complete Python code to implement the following video editing plan:

{plan_text}

Output path: {output_video_path}

Generate a complete, executable Python script that implements this plan. The code should be ready to run and should handle all the operations specified in the plan."""

        # Generate code using LLM
        progress_tracker.add_info("Generating Python code with LLM...")
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        response = llm.invoke(messages)
        generated_code = response.content.strip()
        
        # Extract code if it's wrapped in markdown code blocks
        if generated_code.startswith("```python"):
            # Remove markdown code block markers
            lines = generated_code.split("\n")
            # Remove first line (```python)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            generated_code = "\n".join(lines)
        elif generated_code.startswith("```"):
            # Generic code block
            lines = generated_code.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            generated_code = "\n".join(lines)
        
        # Save generated code to file
        code_path = file_manager.save_text("generated_code.py", generated_code)
        progress_tracker.add_info(f"Generated code saved to: {code_path}")
        
        progress_tracker.set_stage("CODE_GENERATION", {
            "status": "completed",
            "code_path": code_path
        })
        
        return {
            **state,
            "generated_code": generated_code,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
        
    except Exception as e:
        error_msg = f"Error in code generation: {str(e)}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "errors": errors,
            "generated_code": None,
            "progress": progress_tracker.get_progress_dict()
        }

