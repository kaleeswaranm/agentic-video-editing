"""Code Reviewer Agent - Reviews and optionally updates the generated Python code."""

import re
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker
from utils.llm_utils import get_llm_for_coding


def code_reviewer_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for code review.
    
    Uses LLM (GPT-5 for coding) to review the generated Python code for
    correctness, completeness, and quality. Either approves it or provides
    an updated/corrected version.
    """
    file_manager = FileManager()
    progress_tracker = ProgressTracker()
    
    progress_tracker.set_stage("CODE_REVIEW", {})
    
    generated_code = state.get("generated_code")
    reviewed_code = state.get("reviewed_code")  # May exist from previous review
    reviewed_plan = state.get("reviewed_plan")
    execution_result = state.get("execution_result")
    errors = state.get("errors", [])
    
    # Use reviewed_code if available, otherwise use generated_code
    current_code = reviewed_code or generated_code
    
    if not current_code:
        error_msg = "Code not found in state. Run editor agent first."
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        return {
            **state,
            "errors": errors,
            "reviewed_code": None,
            "progress": progress_tracker.get_progress_dict()
        }
    
    try:
        # Initialize LLM for coding
        progress_tracker.add_info("Initializing LLM for code review...")
        llm = get_llm_for_coding()
        
        # Get plan text for reference
        plan_text = ""
        if reviewed_plan:
            plan_text = reviewed_plan.get("plan_text", "")
        
        # Check if this is an error-based review (execution failed)
        is_error_review = execution_result and not execution_result.get("success", True)
        
        if is_error_review:
            # Error-based review mode
            progress_tracker.add_info("Reviewing code to fix execution errors...")
            error_message = execution_result.get("error_message", "")
            stderr = execution_result.get("stderr", "")
            stdout = execution_result.get("stdout", "")
            retry_count = execution_result.get("retry_count", 0)
            
            print("\n" + "=" * 60)
            print("CODE REVIEWER AGENT - Error Review:")
            print("=" * 60)
            print(f"Error Message: {error_message}")
            # print(f"Stderr: {stderr}")
            # print(f"Stdout: {stdout}")
            print("=" * 60 + "\n")

            system_message = """You are an expert Python code reviewer specializing in fixing execution errors. Your task is to fix the code based on execution errors.

The code was executed but failed with errors. Your job is to:
1. Analyze the error message and stderr output
2. Identify the root cause of the failure
3. Fix the code to resolve the error
4. **CRITICAL: Scan the entire code for similar errors** - If you fix an error in one place, check if the same pattern exists elsewhere in the code and fix those proactively. For example:
   - If fixing an import error, check all imports throughout the code
   - If fixing an API usage error, check if the same incorrect API is used elsewhere
   - If fixing a function call error, check if similar function calls exist elsewhere
   - Fix all similar issues to prevent future errors
5. Ensure the fix doesn't break other parts of the code
6. Maintain all functionality from the original plan

**IMPORTANT: If the error is related to missing packages, imports, or library installation:**
- **DO NOT add package installation code** - All required packages are already installed in the environment
- **Use an alternative approach** - If a specific library/function is not available or causing errors, use an alternative method from the available libraries to achieve the same functionality
- **Switch libraries if needed** - If MoviePy doesn't have a feature, use opencv-python, scipy, or other available libraries
- **Fix import statements** - If import fails, check if the import path is correct for the library version, or use an alternative import

Common issues to look for:
- File path errors (missing files, incorrect paths)
- MoviePy 1.0.3 API usage errors (ensure code uses MoviePy 1.0.3 patterns)
- Type errors or incorrect function parameters
- **Missing imports or incorrect import statements** - Check that every function/class used is properly imported
- **Incomplete imports** - Verify all required functions from a module are imported (e.g., if using VideoFileClip, TextClip, CompositeVideoClip, all must be imported)
- **Import errors** - If a specific import fails, use an alternative library or method from available libraries
- **Not using available libraries** - Prefer using MoviePy, NumPy, Pillow, imageio, scipy, opencv-python, pydub, tqdm over custom implementations
- Logic errors in video processing
- Resource cleanup issues

Available libraries in the environment (all already installed):
- MoviePy 1.0.3 (from moviepy.editor import ...) - Primary video editing
- NumPy (import numpy) - Array operations
- Pillow/PIL (from PIL import Image) - Image processing
- imageio (import imageio) - Video file I/O
- scipy (import scipy) - Scientific computing
- opencv-python (import cv2) - Advanced image/video processing
- pydub (from pydub import AudioSegment) - Audio processing
- tqdm (from tqdm import tqdm) - Progress bars

If a specific function or feature is not available in one library, use an alternative from another available library.

Respond with the following format:
CHANGES:
[Brief summary of the changes made and the reasons for each change. Explain what was wrong and how you fixed it.]

UPDATED:
```python
[fixed code that resolves the error]
```

Provide the complete fixed code, not just the changes. The CHANGES section should clearly explain what was fixed and why."""

            human_message = f"""The following code failed during execution. Please fix it based on the error information:

Current Code:
```python
{current_code}
```

Execution Error:
Return Code: {execution_result.get('return_code', 'N/A')}
Error Message: {error_message}

Stderr Output:
{stderr}

Stdout Output (if any):
{stdout}

Retry Attempt: {retry_count + 1}

Original Plan (for reference):
{plan_text}

Please analyze the error and provide a fixed version of the code that resolves the issue. Make sure the fix addresses the root cause and doesn't introduce new errors."""

        else:
            # Initial review mode (existing functionality)
            progress_tracker.add_info("Reviewing code for initial quality check...")
            system_message = """You are an expert Python code reviewer specializing in video editing applications. Your task is to review Python code for correctness, completeness, and quality.

Review the code for:
1. **Correctness**: Syntax is valid, logic is correct, MoviePy usage is proper
2. **Completeness**: All operations from the plan are implemented
3. **Quality**: Error handling, code structure, readability, best practices
4. **Bugs**: Identify any potential runtime errors or logic issues
5. **Values and Parameters**: **CRITICAL** - Verify the code uses the exact values and numerical factors as specified in the plan
   - Check that all numerical values (brightness, contrast, volume, speed, blur amount, etc.) match the plan exactly
   - Verify time ranges (start_time, end_time) match the plan
   - Check that text overlay parameters (font size, position, duration, timing) match the plan
   - Verify audio parameters (volume levels, fade durations) match the plan
   - Check that effect parameters (blur radius, color correction values, etc.) match the plan
   - Ensure file paths match the plan
   - If any values differ from the plan, fix them to match exactly
6. **Imports**: **CRITICAL** - Verify all imports are correct and complete
   - Check that every function, class, or module used in the code is properly imported
   - Verify import statements match the actual usage in the code
   - Ensure MoviePy 1.0.3 imports are correct (from moviepy.editor import ...)
   - Check that all required modules are imported if used (numpy, PIL, imageio, scipy, opencv-python, pydub, tqdm, etc.)
   - If a function is used but not imported, add the correct import statement
   - **Prefer using available libraries**: MoviePy, NumPy, Pillow, imageio, scipy, opencv-python, pydub, tqdm
7. **Library Usage**: Verify the code uses available libraries appropriately
   - MoviePy 1.0.3 for primary video editing operations
   - NumPy for array operations and numerical computations
   - Pillow/PIL for image processing
   - imageio for video file I/O
   - opencv-python (cv2) for advanced image/video processing and effects
   - scipy for scientific computing operations
   - pydub for advanced audio processing if needed
   - tqdm for progress bars on long operations
   - Prefer standard libraries over custom implementations
8. **MoviePy Patterns**: Verify correct usage of MoviePy library functions
   - IMPORTANT: Verify the code uses MoviePy version 1.0.3 API patterns
   - The code should use standard MoviePy 1.0.3 imports (from moviepy.editor import ...)
   - MoviePy 1.0.3 is already installed in the environment, no need for installation code
9. **File Handling**: Check file paths, directory creation, and resource cleanup

If the code is good and complete, respond with:
APPROVED: [brief confirmation message]

If the code has issues or needs improvements, respond with:
UPDATED:
```python
[corrected and improved code]
```

Be thorough but focused. Only update if there are actual issues that need fixing."""

            human_message = f"""Review the following Python code for video editing:

Generated Code:
```python
{current_code}
```

Original Plan (for reference):
{plan_text}

Please review this code against the plan above. Check that all operations from the plan are implemented correctly, the code is syntactically correct, uses MoviePy properly, and follows best practices. Either approve it or provide an updated version if you find any issues."""
        
        # Generate review using LLM
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        response = llm.invoke(messages)
        review_response = response.content.strip()
        
        # Parse response to determine if approved or updated
        if is_error_review:
            # Error-based review always returns updated code
            progress_tracker.add_info("Code updated to fix execution errors")
            
            # Extract changes summary if present
            changes_match = re.search(r'CHANGES:\s*\n(.*?)(?=\nUPDATED:|$)', review_response, re.DOTALL | re.IGNORECASE)
            if changes_match:
                changes_summary = changes_match.group(1).strip()
                print("\n" + "=" * 60)
                print("CODE REVIEWER - Changes Made:")
                print("=" * 60)
                print(changes_summary)
                print("=" * 60 + "\n")
            
            # Extract code from markdown code block
            code_block_match = re.search(r'```python\s*\n(.*?)\n```', review_response, re.DOTALL)
            if code_block_match:
                reviewed_code = code_block_match.group(1).strip()
            else:
                # Try generic code block
                code_block_match = re.search(r'```\s*\n(.*?)\n```', review_response, re.DOTALL)
                if code_block_match:
                    reviewed_code = code_block_match.group(1).strip()
                else:
                    # Fallback: extract everything after "UPDATED:" or "UPDATED:\n"
                    updated_match = re.search(r'UPDATED:?\s*\n?(.*)', review_response, re.DOTALL | re.IGNORECASE)
                    if updated_match:
                        reviewed_code = updated_match.group(1).strip()
                    else:
                        # Last resort: use everything after "UPDATED"
                        reviewed_code = re.sub(r'^UPDATED:?\s*', '', review_response, flags=re.IGNORECASE).strip()
            was_updated = True
        elif review_response.upper().startswith("APPROVED"):
            # Code is approved, use current code
            progress_tracker.add_info("Code approved by reviewer")
            reviewed_code = current_code
            was_updated = False
        elif "UPDATED" in review_response.upper() or "CHANGES" in review_response.upper():
            # Code was updated, extract changes summary and code
            progress_tracker.add_info("Code updated by reviewer")
            
            # Extract changes summary if present
            changes_match = re.search(r'CHANGES:\s*\n(.*?)(?=\nUPDATED:|$)', review_response, re.DOTALL | re.IGNORECASE)
            if changes_match:
                changes_summary = changes_match.group(1).strip()
                print("\n" + "=" * 60)
                print("CODE REVIEWER - Changes Made:")
                print("=" * 60)
                print(changes_summary)
                print("=" * 60 + "\n")
            
            # Extract code from markdown code block
            code_block_match = re.search(r'```python\s*\n(.*?)\n```', review_response, re.DOTALL)
            if code_block_match:
                reviewed_code = code_block_match.group(1).strip()
            else:
                # Try generic code block
                code_block_match = re.search(r'```\s*\n(.*?)\n```', review_response, re.DOTALL)
                if code_block_match:
                    reviewed_code = code_block_match.group(1).strip()
                else:
                    # Fallback: extract everything after "UPDATED:" or "UPDATED:\n"
                    updated_match = re.search(r'UPDATED:?\s*\n?(.*)', review_response, re.DOTALL | re.IGNORECASE)
                    if updated_match:
                        reviewed_code = updated_match.group(1).strip()
                    else:
                        # Last resort: use everything after "UPDATED"
                        reviewed_code = re.sub(r'^UPDATED:?\s*', '', review_response, flags=re.IGNORECASE).strip()
            was_updated = True
        else:
            # Ambiguous response, treat as approved but log warning
            progress_tracker.add_info("Warning: Unclear review response, treating as approved")
            reviewed_code = current_code
            was_updated = False
        
        # Save reviewed code to file
        # First, check if reviewed_code.py exists and rename it to preserve history
        artifacts_dir = Path("artifacts")
        existing_file = artifacts_dir / "reviewed_code.py"
        if existing_file.exists():
            # Find the next available version number
            version = 1
            while (artifacts_dir / f"reviewed_code_{version}.py").exists():
                version += 1
            # Rename the existing file
            old_file_path = artifacts_dir / f"reviewed_code_{version}.py"
            existing_file.rename(old_file_path)
            progress_tracker.add_info(f"Renamed previous reviewed_code.py to reviewed_code_{version}.py")
            print(f"📦 Archived previous code as: reviewed_code_{version}.py")
        
        # Save new reviewed code
        reviewed_code_path = file_manager.save_text("reviewed_code.py", reviewed_code)
        progress_tracker.add_info(f"Reviewed code saved to: {reviewed_code_path}")
        print(f"\n✅ Reviewed code saved to: {reviewed_code_path}")
        # print(f"📝 Code length: {len(reviewed_code)} characters")
        # print(f"📝 First 100 chars: {reviewed_code[:100]}...\n")
        
        progress_tracker.set_stage("CODE_REVIEW", {
            "status": "completed",
            "was_updated": was_updated,
            "code_path": reviewed_code_path
        })
        
        return {
            **state,
            "reviewed_code": reviewed_code,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
        
    except Exception as e:
        error_msg = f"Error in code review: {str(e)}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "errors": errors,
            "reviewed_code": None,
            "progress": progress_tracker.get_progress_dict()
        }

