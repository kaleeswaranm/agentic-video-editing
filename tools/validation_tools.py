"""Validation utility functions."""

from typing import List, Tuple, Dict, Any
import pandas as pd
from pathlib import Path

from tools.file_tools import check_file_exists, is_video_file, is_audio_file


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
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    errors.append(f"Column '{col}' contains non-numeric values")
            except (ValueError, TypeError):
                errors.append(f"Column '{col}' must contain numeric values")
    
    return len(errors) == 0, errors


def validate_time_ranges(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate time ranges (start_time < end_time, both >= 0)."""
    errors = []
    
    if 'start_time' in df.columns and 'end_time' in df.columns:
        for idx, row in df.iterrows():
            start = row.get('start_time')
            end = row.get('end_time')
            
            if pd.isna(start) or pd.isna(end):
                errors.append(f"Row {idx + 1}: Missing start_time or end_time")
                continue
            
            if start < 0:
                errors.append(f"Row {idx + 1}: start_time ({start}) must be >= 0")
            
            if end < 0:
                errors.append(f"Row {idx + 1}: end_time ({end}) must be >= 0")
            
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
    """Validate that video files exist at specified paths."""
    errors = []
    warnings = []
    
    if 'video_path' in df.columns:
        for idx, row in df.iterrows():
            video_path = row.get('video_path')
            if pd.isna(video_path) or not video_path:
                errors.append(f"Row {idx + 1}: video_path is empty")
                continue
            
            if not check_file_exists(str(video_path)):
                errors.append(f"Row {idx + 1}: Video file not found: {video_path}")
            elif not is_video_file(str(video_path)):
                warnings.append(f"Row {idx + 1}: File may not be a video: {video_path}")
    
    return len(errors) == 0, errors, warnings


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
    
    # Audio path validation
    valid, errors = validate_audio_paths_in_descriptions(df)
    all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors, all_warnings

