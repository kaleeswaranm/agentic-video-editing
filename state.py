"""LangGraph state definition for video editing workflow."""

from typing import TypedDict, List, Optional, Any, Dict


class VideoEditingState(TypedDict):
    """State schema for the video editing workflow."""
    
    # Input
    csv_path: str
    
    # Intermediate artifacts
    validated_csv: Optional[Dict[str, Any]]
    editing_plan: Optional[Dict[str, Any]]
    reviewed_plan: Optional[Dict[str, Any]]
    generated_code: Optional[str]
    reviewed_code: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    output_video_path: Optional[str]
    validation_result: Optional[Dict[str, Any]]
    
    # Workflow control
    progress: Dict[str, Any]  # Current stage, progress info
    errors: List[str]  # List of errors encountered
    feedback: Optional[str]  # Feedback from review agents

