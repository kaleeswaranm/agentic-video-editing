"""File operation tools for agents."""

from pathlib import Path
from typing import List, Optional


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists at the given path."""
    path = Path(filepath)
    if path.is_absolute():
        return path.exists()
    # Try relative to current working directory
    return path.exists()


def get_file_extension(filepath: str) -> str:
    """Get file extension from filepath."""
    return Path(filepath).suffix.lower()


def is_video_file(filepath: str) -> bool:
    """Check if file is a video file based on extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    return get_file_extension(filepath) in video_extensions


def is_audio_file(filepath: str) -> bool:
    """Check if file is an audio file based on extension."""
    audio_extensions = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma'}
    return get_file_extension(filepath) in audio_extensions


def get_absolute_path(filepath: str) -> Optional[str]:
    """Get absolute path for a file."""
    path = Path(filepath)
    if path.is_absolute():
        return str(path) if path.exists() else None
    abs_path = path.resolve()
    return str(abs_path) if abs_path.exists() else None


def list_files_in_directory(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """List files in a directory, optionally filtered by extensions."""
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    
    files = []
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(str(file_path))
    
    return files

