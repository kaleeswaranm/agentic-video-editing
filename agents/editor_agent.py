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
    file_manager = FileManager()
    progress_tracker = ProgressTracker()
    
    progress_tracker.set_stage("CODE_GENERATION", {})
    
    reviewed_plan = state.get("reviewed_plan")
    output_video_path = state.get("output_video_path", "artifacts/output_video.mp4")
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
7. Create intermediate directories if needed (e.g., artifacts/temp/)
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
- **Prefer using available libraries**: MoviePy, NumPy, Pillow, imageio, scipy, opencv-python, pydub, tqdm
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

