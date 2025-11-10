# Agentic Video Editing Tool

AI-powered video editing tool  to automatically create edited videos with transitions, effects, text overlays, and audio mixing.

Find a edit produced using this tool here - `artifacts/output_video.mp4`

## Installation

```bash
pip install -r requirements.txt
```

**Note:** Requires `exiftool` for phone video orientation detection:
- macOS: `brew install exiftool`
- Linux: `apt-get install libimage-exiftool-perl` or `yum install perl-Image-ExifTool`

## Quick Start

```bash
python main.py video_editing_config.csv --artifacts-dir artifacts
```

## CSV Format

Required columns:
- `order` - Clip sequence number
- `video_path` - Path to video/image file
- `start_time` - Start time in seconds
- `end_time` - End time in seconds (use `-1` for "till end of clip", optional for images)
- `transition` - Transition type (fade in/out, etc.) with any specifications
- `effects` - Video effects (speed, brightness, blur, etc.) with any specifications
- `overlay_text` - Text to overlay
- `overlay_description` - Text position, timing, styling, duration, etc
- `audio_description` - Audio mode (keep/replace/mix), volume, external audio path, transitions, etc
- `additional_operations` - Additional ops (rotate, scale, crop), image duration (e.g., "display for 0.5s"), or any other operations/tasks

Check the available config csv files for reference.

## Configuration

Environment variables (optional, can also use CLI args. Preferred create a .env file):
- `OPENAI_API_KEY` - Your openai API key
- `LLM_PROVIDER` - openai
- `LLM_MODEL` - Model name (default: gpt-4o-mini)
- `PLANNING_MODEL` - Model for planning tasks (default: gpt-5)
- `CODING_MODEL` - Model for code generation (default: gpt-5)

CLI options:
- `--llm-provider` - LLM provider
- `--llm-model` - LLM model
- `--planning-model` - Planning model
- `--coding-model` - Coding model
- `--artifacts-dir` - Directory for artifacts (default: artifacts)
- `--output` - Output video path

## Output

- Final video: `--output` path (default: `artifacts/output_video.mp4`)
- Artifacts: Plans, reviewed plans, generated code, reviewed code and progress saved in `--artifacts-dir`.

## Workflow

1. **CSV Validation** - Validates CSV structure and file paths
2. **Planning** - LLM generates editing plan from CSV
3. **Plan Review** - LLM reviews and refines the plan
4. **Code Generation** - LLM generates Python code to execute the plan
5. **Code Review** - LLM reviews and fixes code
6. **Execution** - Code runs and feedback loop to Code Review for debugging (up to 15 retries)
