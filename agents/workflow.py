"""LangGraph workflow for video editing orchestration."""

from langgraph.graph import StateGraph, END
from typing import Literal

from state import VideoEditingState
from agents.csv_validator_agent import csv_validator_node
from agents.planning_agent import planning_node
from agents.plan_reviewer_agent import plan_reviewer_node
from agents.editor_agent import editor_node
from agents.code_reviewer_agent import code_reviewer_node
from agents.execution_agent import execution_node, MAX_RETRIES


def should_continue_after_validation(state: VideoEditingState) -> Literal["planning", "error"]:
    """Determine next node based on validation result."""
    if state.get("errors") and len(state.get("errors", [])) > 0:
        return "error"
    if state.get("validated_csv") is not None:
        return "planning"
    return "error"


def should_continue_after_planning(state: VideoEditingState) -> Literal["plan_reviewer", "error"]:
    """Determine next node based on planning result."""
    if state.get("errors") and len(state.get("errors", [])) > 0:
        return "error"
    if state.get("editing_plan") is not None:
        return "plan_reviewer"
    return "error"


def should_continue_after_plan_review(state: VideoEditingState) -> Literal["editor", "error"]:
    """Determine next node based on plan review result."""
    if state.get("errors") and len(state.get("errors", [])) > 0:
        return "error"
    if state.get("reviewed_plan") is not None:
        return "editor"
    return "error"


def should_continue_after_editing(state: VideoEditingState) -> Literal["code_reviewer", "error"]:
    """Determine next node based on code generation result."""
    if state.get("errors") and len(state.get("errors", [])) > 0:
        return "error"
    if state.get("generated_code") is not None:
        return "code_reviewer"
    return "error"


def should_continue_after_code_review(state: VideoEditingState) -> Literal["execution", "error"]:
    """Determine next node based on code review result."""
    if state.get("errors") and len(state.get("errors", [])) > 0:
        return "error"
    if state.get("reviewed_code") is not None:
        return "execution"
    return "error"


def should_continue_after_execution(state: VideoEditingState) -> Literal["video_validator", "code_reviewer", "error"]:
    """Determine next node based on execution result."""
    if state.get("errors") and len(state.get("errors", [])) > 0:
        return "error"
    
    execution_result = state.get("execution_result")
    if not execution_result:
        return "error"
    
    if execution_result.get("success", False):
        # Execution succeeded, proceed to video validation
        return "video_validator"
    
    # Execution failed, check retry count
    retry_count = execution_result.get("retry_count", 0)
    if retry_count < MAX_RETRIES:
        # Retry by going back to code reviewer
        return "code_reviewer"
    else:
        # Max retries exceeded
        return "error"


def create_workflow() -> StateGraph:
    """Create and compile the LangGraph workflow."""
    
    # Create the graph
    workflow = StateGraph(VideoEditingState)
    
    # Add nodes
    workflow.add_node("csv_validator", csv_validator_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("plan_reviewer", plan_reviewer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("code_reviewer", code_reviewer_node)
    workflow.add_node("execution", execution_node)
    
    # Placeholder nodes for future implementation
    def video_validator_node(state: VideoEditingState) -> VideoEditingState:
        """Placeholder for video validator agent."""
        return {**state, "validation_result": None}
    
    workflow.add_node("video_validator", video_validator_node)
    
    # Set entry point
    workflow.set_entry_point("csv_validator")
    
    # Add edges from csv_validator
    workflow.add_conditional_edges(
        "csv_validator",
        should_continue_after_validation,
        {
            "planning": "planning",
            "error": END
        }
    )
    
    # Add edges from planning
    workflow.add_conditional_edges(
        "planning",
        should_continue_after_planning,
        {
            "plan_reviewer": "plan_reviewer",
            "error": END
        }
    )
    
    # Add edges from plan_reviewer
    workflow.add_conditional_edges(
        "plan_reviewer",
        should_continue_after_plan_review,
        {
            "editor": "editor",
            "error": END
        }
    )
    
    # Add edges from editor
    workflow.add_conditional_edges(
        "editor",
        should_continue_after_editing,
        {
            "code_reviewer": "code_reviewer",
            "error": END
        }
    )
    
    # Add edges from code_reviewer
    workflow.add_conditional_edges(
        "code_reviewer",
        should_continue_after_code_review,
        {
            "execution": "execution",
            "error": END
        }
    )
    
    # Add edges from execution (feedback loop)
    workflow.add_conditional_edges(
        "execution",
        should_continue_after_execution,
        {
            "video_validator": "video_validator",
            "code_reviewer": "code_reviewer",  # Feedback loop for retries
            "error": END
        }
    )
    
    # For now, video_validator goes to END (will be updated in future steps)
    workflow.add_edge("video_validator", END)
    
    return workflow.compile()


def get_workflow() -> StateGraph:
    """Get the compiled workflow."""
    return create_workflow()

