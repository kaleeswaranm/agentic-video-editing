"""Execution Agent - Executes the reviewed Python code."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker

# Configuration
MAX_RETRIES = 15
EXECUTION_TIMEOUT = 3600  # 1 hour in seconds


def execution_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for code execution.
    
    Executes the reviewed Python code using subprocess and captures
    output, errors, and execution results.
    """
    file_manager = FileManager()
    progress_tracker = ProgressTracker()
    
    progress_tracker.set_stage("EXECUTION", {})
    
    reviewed_code = state.get("reviewed_code")
    output_video_path = state.get("output_video_path", "artifacts/output_video.mp4")
    execution_result = state.get("execution_result", {})
    errors = state.get("errors", [])
    
    if not reviewed_code:
        error_msg = "Reviewed code not found in state. Run code reviewer first."
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        return {
            **state,
            "errors": errors,
            "execution_result": None,
            "progress": progress_tracker.get_progress_dict()
        }
    
    # Get retry count from previous execution result or initialize to 0
    retry_count = execution_result.get("retry_count", 0) if execution_result else 0
    
    try:
        # Save code to file for execution
        code_file_path = file_manager.save_text("reviewed_code.py", reviewed_code)
        attempt_number = retry_count + 1
        progress_tracker.add_info(f"Executing code from: {code_file_path}")
        progress_tracker.add_info(f"Attempt {attempt_number} of {MAX_RETRIES}")
        
        # Execute code using subprocess
        start_time = time.time()
        
        progress_tracker.add_info("Starting code execution...")
        process = subprocess.Popen(
            [sys.executable, code_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Capture output in real-time (optional, for long operations)
        stdout_lines = []
        stderr_lines = []
        
        try:
            # Wait for process with timeout
            stdout, stderr = process.communicate(timeout=EXECUTION_TIMEOUT)
            stdout_lines = stdout.split('\n') if stdout else []
            stderr_lines = stderr.split('\n') if stderr else []
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            stdout_lines = stdout.split('\n') if stdout else []
            stderr_lines = stderr.split('\n') if stderr else []
            raise TimeoutError(f"Code execution timed out after {EXECUTION_TIMEOUT} seconds")
        
        execution_time = time.time() - start_time
        return_code = process.returncode
        
        stdout_text = '\n'.join(stdout_lines)
        stderr_text = '\n'.join(stderr_lines)
        
        # Check if execution was successful
        # First verify output video exists
        output_path = Path(output_video_path)
        video_exists = output_path.exists()
        
        if not video_exists:
            # Check if code produced video at different location
            # Look for common output locations
            possible_paths = [
                Path("artifacts/output_video.mp4"),
                Path(output_video_path),
                Path("output_video.mp4"),
            ]
            found_path = None
            for path in possible_paths:
                if path.exists():
                    found_path = str(path.absolute())
                    break
            
            if found_path:
                progress_tracker.add_info(f"Output video found at: {found_path}")
                output_video_path = found_path
                video_exists = True
        
        # Check for error messages in stderr/stdout even if return_code == 0
        # (code might catch exceptions and exit with 0)
        has_error_messages = False
        error_keywords = ["Error", "error", "Exception", "Traceback", "failed", "Failed"]
        if stderr_text:
            has_error_messages = any(keyword in stderr_text for keyword in error_keywords)
        if stdout_text and not has_error_messages:
            has_error_messages = any(keyword in stdout_text for keyword in error_keywords)
        
        # Consider execution successful only if:
        # 1. return_code == 0 AND
        # 2. video file exists AND
        # 3. no error messages in output
        if return_code == 0 and video_exists and not has_error_messages:
            # Execution succeeded
            progress_tracker.add_info(f"Code executed successfully in {execution_time:.2f} seconds")
            progress_tracker.add_info(f"Output video created: {output_video_path}")
            
            execution_result = {
                "success": True,
                "return_code": return_code,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "execution_time": execution_time,
                "output_video_path": output_video_path,
                "retry_count": retry_count
            }
            
            progress_tracker.set_stage("EXECUTION", {
                "status": "completed",
                "success": True,
                "execution_time": execution_time,
                "output_video_path": output_video_path
            })
            
            return {
                **state,
                "execution_result": execution_result,
                "output_video_path": output_video_path,
                "errors": errors,
                "progress": progress_tracker.get_progress_dict()
            }
        
        # Execution failed - either non-zero return code, missing video, or error messages
        # Increment retry count
        retry_count += 1
        
        # Extract actual error messages from output
        actual_errors = []
        if stderr_text:
            error_lines = [line.strip() for line in stderr_lines if line.strip() and any(keyword in line for keyword in ["Error", "error", "Exception", "Traceback", "failed", "Failed"])]
            if error_lines:
                actual_errors.extend(error_lines[-5:])  # Last 5 error lines from stderr
        if stdout_text:
            error_lines = [line.strip() for line in stdout_lines if line.strip() and any(keyword in line for keyword in ["Error", "error", "Exception", "Traceback", "failed", "Failed"])]
            if error_lines:
                actual_errors.extend(error_lines[-5:])  # Last 5 error lines from stdout
        
        # Extract the actual error message to send to code reviewer
        if actual_errors:
            # Use actual error messages found in output
            error_message = "\n".join(actual_errors[-3:])  # Last 3 error lines
        elif stderr_text:
            # Fallback: use last meaningful line of stderr
            error_lines = [line.strip() for line in stderr_lines if line.strip()]
            if error_lines:
                error_message = error_lines[-1]
            else:
                error_message = "Execution failed - check stderr for details"
        elif stdout_text:
            # Fallback: check stdout for any error-like messages
            error_lines = [line.strip() for line in stdout_lines if line.strip() and ("error" in line.lower() or "failed" in line.lower())]
            if error_lines:
                error_message = error_lines[-1]
            else:
                error_message = "Execution failed - output video file was not created"
        else:
            if return_code == 0:
                error_message = "Output video file was not created"
            else:
                error_message = f"Code execution failed with return code {return_code}"
        
        progress_tracker.add_error(f"Execution failed (attempt {retry_count}/{MAX_RETRIES}): {error_message}")
        # progress_tracker.add_info(f"Stderr output: {stderr_text[:500]}...")  # First 500 chars
        
        execution_result = {
            "success": False,
            "return_code": return_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "execution_time": execution_time,
            "error_message": error_message,
            "retry_count": retry_count
        }
        
        progress_tracker.set_stage("EXECUTION", {
            "status": "failed",
            "success": False,
            "return_code": return_code,
            "error_message": error_message,
            "retry_count": retry_count
        })
        
        return {
            **state,
            "execution_result": execution_result,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
        
    except TimeoutError as e:
        error_msg = str(e)
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        
        execution_result = {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": error_msg,
            "execution_time": EXECUTION_TIMEOUT,
            "error_message": error_msg,
            "retry_count": retry_count
        }
        
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "execution_result": execution_result,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
    
    except Exception as e:
        error_msg = f"Error during code execution: {str(e)}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        
        execution_result = {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": 0,
            "error_message": error_msg,
            "retry_count": retry_count
        }
        
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "execution_result": execution_result,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }

