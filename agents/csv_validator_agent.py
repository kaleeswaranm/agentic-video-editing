"""CSV Validator Agent - Validates CSV input file."""

import pandas as pd
from typing import Dict, Any

from state import VideoEditingState
from utils.file_manager import FileManager
from utils.progress_tracker import ProgressTracker
from tools.validation_tools import validate_csv_complete


def csv_validator_node(state: VideoEditingState) -> VideoEditingState:
    """
    LangGraph node function for CSV validation.
    
    Validates the CSV file structure, data types, file paths, and time ranges.
    Saves validated CSV data to artifacts.
    """
    file_manager = FileManager()
    progress_tracker = ProgressTracker()
    
    progress_tracker.set_stage("CSV_VALIDATION", {"csv_path": state.get("csv_path", "")})
    
    csv_path = state.get("csv_path", "")
    errors = state.get("errors", [])
    
    if not csv_path:
        error_msg = "CSV path not provided in state"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        return {
            **state,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
    
    try:
        # Read CSV file
        progress_tracker.add_info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Validate CSV
        progress_tracker.add_info("Validating CSV structure and data...")
        is_valid, validation_errors, warnings = validate_csv_complete(df)
        
        # Add warnings as info
        for warning in warnings:
            progress_tracker.add_info(f"Warning: {warning}")
        
        # Add errors
        if validation_errors:
            errors.extend(validation_errors)
            for error in validation_errors:
                progress_tracker.add_error(error)
        
        if not is_valid:
            progress_tracker.set_stage("ERROR", {"errors": validation_errors})
            return {
                **state,
                "errors": errors,
                "validated_csv": None,
                "progress": progress_tracker.get_progress_dict()
            }
        
        # Convert DataFrame to dictionary for JSON serialization
        # Replace NaN values with None for proper JSON serialization
        df_clean = df.where(pd.notna(df), None)
        validated_data = {
            "csv_path": csv_path,
            "total_clips": len(df),
            "clips": df_clean.to_dict('records')
        }
        
        # Save validated CSV to artifacts
        artifact_path = file_manager.save_json("validated_csv.json", validated_data)
        progress_tracker.add_info(f"Validated CSV saved to: {artifact_path}")
        
        progress_tracker.set_stage("CSV_VALIDATION", {
            "status": "completed",
            "total_clips": len(df),
            "artifact_path": artifact_path
        })
        
        return {
            **state,
            "validated_csv": validated_data,
            "errors": errors,
            "progress": progress_tracker.get_progress_dict()
        }
        
    except FileNotFoundError:
        error_msg = f"CSV file not found: {csv_path}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "errors": errors,
            "validated_csv": None,
            "progress": progress_tracker.get_progress_dict()
        }
    
    except pd.errors.EmptyDataError:
        error_msg = f"CSV file is empty: {csv_path}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "errors": errors,
            "validated_csv": None,
            "progress": progress_tracker.get_progress_dict()
        }
    
    except Exception as e:
        error_msg = f"Error validating CSV: {str(e)}"
        errors.append(error_msg)
        progress_tracker.add_error(error_msg)
        progress_tracker.set_stage("ERROR", {"error": error_msg})
        return {
            **state,
            "errors": errors,
            "validated_csv": None,
            "progress": progress_tracker.get_progress_dict()
        }

