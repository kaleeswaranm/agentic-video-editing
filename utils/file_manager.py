"""File management utilities for saving/loading artifacts."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class FileManager:
    """Manages artifact file operations."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """Initialize FileManager with artifacts directory."""
        self.artifacts_dir = Path(artifacts_dir)
        self.ensure_artifacts_dir()
    
    def ensure_artifacts_dir(self) -> None:
        """Ensure artifacts directory exists."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, filename: str, data: Dict[str, Any]) -> str:
        """Save data as JSON file in artifacts directory."""
        filepath = self.artifacts_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(filepath)
    
    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load JSON file from artifacts directory."""
        filepath = self.artifacts_dir / filename
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_text(self, filename: str, content: str) -> str:
        """Save text content to file in artifacts directory."""
        filepath = self.artifacts_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(filepath)
    
    def load_text(self, filename: str) -> Optional[str]:
        """Load text content from file in artifacts directory."""
        filepath = self.artifacts_dir / filename
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def file_exists(self, filepath: str) -> bool:
        """Check if a file exists (absolute or relative to workspace)."""
        path = Path(filepath)
        if path.is_absolute():
            return path.exists()
        # Try relative to current working directory
        return path.exists()
    
    def get_absolute_path(self, filepath: str) -> Optional[str]:
        """Get absolute path for a file (relative or absolute)."""
        path = Path(filepath)
        if path.is_absolute():
            return str(path) if path.exists() else None
        # Try relative to current working directory
        abs_path = path.resolve()
        return str(abs_path) if abs_path.exists() else None

