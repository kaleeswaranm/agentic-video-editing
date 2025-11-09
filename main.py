"""Main entry point for video editing workflow."""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from agents.workflow import get_workflow
from agents.execution_agent import MAX_RETRIES
from state import VideoEditingState

# Load environment variables
load_dotenv()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Video Editing Tool - Processes CSV and creates edited video"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/output_video.mp4",
        help="Output video path (default: artifacts/output_video.mp4)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=os.getenv("LLM_PROVIDER", "openai"),
        choices=["openai", "anthropic", "ollama"],
        help="LLM provider to use (default: openai, or from LLM_PROVIDER env var)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        help="LLM model to use (default: gpt-4o-mini, or from LLM_MODEL env var)"
    )
    parser.add_argument(
        "--planning-model",
        type=str,
        default=os.getenv("PLANNING_MODEL", "gpt-5"),
        help="LLM model for planning/reviewing tasks (default: gpt-5, or from PLANNING_MODEL env var)"
    )
    parser.add_argument(
        "--coding-model",
        type=str,
        default=os.getenv("CODING_MODEL", "gpt-5"),
        help="LLM model for coding/code review tasks (default: gpt-5, or from CODING_MODEL env var)"
    )
    
    args = parser.parse_args()
    
    # Store LLM config in environment for agents to access
    os.environ["LLM_PROVIDER"] = args.llm_provider
    os.environ["LLM_MODEL"] = args.llm_model
    os.environ["PLANNING_MODEL"] = args.planning_model
    os.environ["CODING_MODEL"] = args.coding_model
    
    # Validate CSV path exists
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Initialize state
    initial_state: VideoEditingState = {
        "csv_path": str(csv_path.absolute()),
        "validated_csv": None,
        "editing_plan": None,
        "reviewed_plan": None,
        "generated_code": None,
        "reviewed_code": None,
        "execution_result": None,
        "output_video_path": args.output,
        "validation_result": None,
        "progress": {},
        "errors": [],
        "feedback": None
    }
    
    print("=" * 60)
    print("Agentic Video Editing Tool")
    print("=" * 60)
    print(f"CSV File: {csv_path.absolute()}")
    print(f"Output: {args.output}")
    print(f"LLM Provider: {args.llm_provider}")
    print(f"Planning Model: {args.planning_model}")
    print(f"Coding Model: {args.coding_model}")
    print("=" * 60)
    print()
    
    try:
        # Get workflow
        workflow = get_workflow()
        
        # Run workflow
        print("Starting workflow...\n")
        final_state = workflow.invoke(initial_state)
        
        # Check for errors or failed execution
        has_errors = final_state.get("errors") and len(final_state.get("errors", [])) > 0
        execution_result = final_state.get("execution_result")
        execution_failed = execution_result and not execution_result.get("success", True)
        max_retries_exceeded = execution_result and execution_result.get("retry_count", 0) >= MAX_RETRIES
        
        if has_errors or (execution_failed and max_retries_exceeded):
            print("\n" + "=" * 60)
            print("Workflow completed with errors:")
            print("=" * 60)
            if has_errors:
                for error in final_state["errors"]:
                    print(f"  - {error}")
            if execution_failed and max_retries_exceeded:
                print(f"  - Execution failed after {execution_result.get('retry_count', 0)} attempts")
                if execution_result.get("error_message"):
                    print(f"  - Last error: {execution_result['error_message']}")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("Workflow completed successfully!")
        print("=" * 60)
        
        # Display final state summary
        if final_state.get("validated_csv"):
            print(f"Validated clips: {final_state['validated_csv'].get('total_clips', 0)}")
        
        current_stage = final_state.get("progress", {}).get("current_stage", "Unknown")
        print(f"Final stage: {current_stage}")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

