"""Progress tracking utilities for workflow stages."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ProgressTracker:
    """Tracks progress through workflow stages."""
    
    STAGES = {
        "CSV_VALIDATION": "Validating CSV file",
        "PLANNING": "Creating editing plan",
        "PLAN_REVIEW": "Reviewing plan",
        "CODE_GENERATION": "Generating code",
        "CODE_REVIEW": "Reviewing code",
        "EXECUTION": "Executing code",
        "VIDEO_VALIDATION": "Validating output video",
        "COMPLETED": "Workflow completed",
        "ERROR": "Error occurred"
    }
    
    def __init__(self, progress_file: str = "artifacts/progress.json"):
        """Initialize ProgressTracker."""
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_stage = None
        self.stage_history = []
    
    def set_stage(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Set current stage and log progress."""
        timestamp = datetime.now().isoformat()
        stage_info = {
            "stage": stage,
            "description": self.STAGES.get(stage, stage),
            "timestamp": timestamp,
            "details": details or {}
        }
        
        self.current_stage = stage
        self.stage_history.append(stage_info)
        self._save_progress()
        self._display_progress(stage_info)
    
    def add_error(self, error: str) -> None:
        """Add error to progress log."""
        error_info = {
            "type": "error",
            "message": error,
            "timestamp": datetime.now().isoformat(),
            "stage": self.current_stage
        }
        self.stage_history.append(error_info)
        self._save_progress()
        print(f"❌ Error: {error}")
    
    def add_info(self, message: str) -> None:
        """Add info message to progress log."""
        info = {
            "type": "info",
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "stage": self.current_stage
        }
        self.stage_history.append(info)
        self._save_progress()
        print(f"ℹ️  {message}")
    
    def _save_progress(self) -> None:
        """Save progress to file."""
        progress_data = {
            "current_stage": self.current_stage,
            "history": self.stage_history
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
    
    def _display_progress(self, stage_info: Dict[str, Any]) -> None:
        """Display progress to console."""
        print(f"\n📊 Stage: {stage_info['description']}")
        if stage_info.get('details'):
            for key, value in stage_info['details'].items():
                print(f"   {key}: {value}")
    
    def get_progress_dict(self) -> Dict[str, Any]:
        """Get progress as dictionary for state."""
        return {
            "current_stage": self.current_stage,
            "stage_description": self.STAGES.get(self.current_stage, "") if self.current_stage else "",
            "history": self.stage_history
        }
    
    def load_progress(self) -> Optional[Dict[str, Any]]:
        """Load progress from file."""
        if not self.progress_file.exists():
            return None
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)

