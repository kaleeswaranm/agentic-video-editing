"""Validation utility functions."""

from typing import List, Tuple, Dict, Any
import pandas as pd
from pathlib import Path

from tools.file_tools import check_file_exists, is_video_file, is_audio_file, is_image_file


REQUIRED_COLUMNS = [
    "order",
    "video_path",
    "start_time",
    "end_time",
    "transition",
    "effects",
    "overlay_text",
    "overlay_description",
    "audio_description",
    "additional_operations"
]


def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate CSV has all required columns."""
    errors = []
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    return len(errors) == 0, errors


def validate_data_types(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate data types of columns."""
    errors = []
    
    # Validate order column
    if 'order' in df.columns:
        try:
            df['order'] = df['order'].astype(int)
        except (ValueError, TypeError):
            errors.append("Column 'order' must contain integers")
    
    # Validate start_time and end_time
    # Note: Image files don't need start_time/end_time (they only need duration in additional_operations)
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Check for NaN values, but allow NaN for image file rows
                if df[col].isna().any():
                    # Check if NaN is only in image file rows
                    if 'video_path' in df.columns:
                        nan_mask = df[col].isna()
                        # Check which rows with NaN are image files
                        image_mask = df['video_path'].apply(
                            lambda x: is_image_file(str(x)) if pd.notna(x) and x else False
                        )
                        # Only report error if there are NaN values in non-image rows
                        non_image_nan = nan_mask & ~image_mask
                        if non_image_nan.any():
                            errors.append(f"Column '{col}' contains non-numeric values")
                    else:
                        # No video_path column, check all NaN
                        errors.append(f"Column '{col}' contains non-numeric values")
            except (ValueError, TypeError):
                errors.append(f"Column '{col}' must contain numeric values")
    
    return len(errors) == 0, errors


def validate_time_ranges(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate time ranges (start_time < end_time, both >= 0, or end_time = -1 for 'till end')."""
    errors = []
    
    if 'start_time' in df.columns and 'end_time' in df.columns:
        for idx, row in df.iterrows():
            start = row.get('start_time')
            end = row.get('end_time')
            video_path = row.get('video_path', '')
            
            # Skip validation for image files (they don't need time ranges, only duration)
            if pd.notna(video_path) and is_image_file(str(video_path)):
                continue
            
            if pd.isna(start) or pd.isna(end):
                errors.append(f"Row {idx + 1}: Missing start_time or end_time")
                continue
            
            # Validate start_time
            if start < 0:
                errors.append(f"Row {idx + 1}: start_time ({start}) must be >= 0")
            
            # Handle special case: end_time = -1 means "till end of clip"
            if end == -1:
                # -1 is valid, skip end_time >= 0 check and start < end check
                continue
            
            # Validate end_time (only if not -1)
            if end < 0:
                errors.append(f"Row {idx + 1}: end_time ({end}) must be >= 0 or -1 (for 'till end')")
            
            # Validate start < end (only if end is not -1)
            if start >= end:
                errors.append(
                    f"Row {idx + 1}: start_time ({start}) must be < end_time ({end})"
                )
    
    return len(errors) == 0, errors


def validate_order_sequence(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate order sequence is consecutive starting from 1."""
    errors = []
    
    if 'order' in df.columns:
        orders = sorted(df['order'].unique())
        expected_orders = list(range(1, len(df) + 1))
        
        if orders != expected_orders:
            errors.append(
                f"Order sequence must be consecutive starting from 1. "
                f"Found: {orders}, Expected: {expected_orders}"
            )
    
    return len(errors) == 0, errors


def validate_file_paths(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that video or image files exist at specified paths."""
    errors = []
    warnings = []
    
    if 'video_path' in df.columns:
        for idx, row in df.iterrows():
            video_path = row.get('video_path')
            if pd.isna(video_path) or not video_path:
                errors.append(f"Row {idx + 1}: video_path is empty")
                continue
            
            if not check_file_exists(str(video_path)):
                errors.append(f"Row {idx + 1}: File not found: {video_path}")
            elif is_video_file(str(video_path)):
                # Valid video file
                pass
            elif is_image_file(str(video_path)):
                # Valid image file - will be validated for duration in validate_image_durations
                pass
            else:
                warnings.append(f"Row {idx + 1}: File may not be a video or image: {video_path}")
    
    return len(errors) == 0, errors, warnings


def validate_image_durations(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that image files have duration specified in additional_operations."""
    errors = []
    
    if 'video_path' in df.columns and 'additional_operations' in df.columns:
        for idx, row in df.iterrows():
            video_path = row.get('video_path')
            additional_ops = str(row.get('additional_operations', ''))
            
            if pd.isna(video_path) or not video_path:
                continue
            
            # Check if this is an image file
            if is_image_file(str(video_path)):
                # Try to extract duration from additional_operations
                import re
                # Look for patterns like "for X seconds", "duration: X", "display for X seconds", "for 0.5s", etc.
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
                
                duration_found = False
                duration_value = None
                
                for pattern in duration_patterns:
                    match = re.search(pattern, additional_ops, re.IGNORECASE)
                    if match:
                        try:
                            duration_value = float(match.group(1))
                            if duration_value > 0:
                                duration_found = True
                                break
                        except (ValueError, IndexError):
                            continue
                
                if not duration_found:
                    errors.append(
                        f"Row {idx + 1}: Image file '{video_path}' requires duration specification "
                        f"in additional_operations (e.g., 'display for 0.2 seconds')"
                    )
                elif duration_value is not None and duration_value <= 0:
                    errors.append(
                        f"Row {idx + 1}: Image file '{video_path}' duration must be positive, "
                        f"found: {duration_value}"
                    )
    
    return len(errors) == 0, errors


def validate_audio_paths_in_descriptions(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate audio file paths mentioned in audio_description."""
    errors = []
    
    if 'audio_description' in df.columns:
        for idx, row in df.iterrows():
            audio_desc = str(row.get('audio_description', ''))
            if 'replace audio with' in audio_desc.lower():
                # Extract file path from description
                # Simple extraction - look for path-like strings
                import re
                # Look for paths that look like file paths
                paths = re.findall(r'[\w/\\]+\.(mp3|wav|aac|flac|ogg|m4a|wma)', audio_desc, re.IGNORECASE)
                for path in paths:
                    # Try to find the full path in the description
                    full_path_match = re.search(rf'[\w/\\]+{re.escape(path)}', audio_desc)
                    if full_path_match:
                        audio_path = full_path_match.group(0)
                        if not check_file_exists(audio_path):
                            errors.append(
                                f"Row {idx + 1}: Audio file not found in audio_description: {audio_path}"
                            )
    
    return len(errors) == 0, errors


def validate_csv_complete(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Run all validations and return results."""
    all_errors = []
    all_warnings = []
    
    # Structure validation
    valid, errors = validate_csv_structure(df)
    all_errors.extend(errors)
    if not valid:
        return False, all_errors, all_warnings
    
    # Data type validation
    valid, errors = validate_data_types(df)
    all_errors.extend(errors)
    
    # Time range validation
    valid, errors = validate_time_ranges(df)
    all_errors.extend(errors)
    
    # Order sequence validation
    valid, errors = validate_order_sequence(df)
    all_errors.extend(errors)
    
    # File path validation
    valid, errors, warnings = validate_file_paths(df)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    # Image duration validation (for image files)
    valid, errors = validate_image_durations(df)
    all_errors.extend(errors)
    
    # Audio path validation
    valid, errors = validate_audio_paths_in_descriptions(df)
    all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors, all_warnings

