# SAM3 Annotation Tool - Copilot Instructions

## Project Overview
This is a Python-based image annotation tool using the **SAM3 (Segment Anything Model 3)** model, built with a **Gradio** web interface.

## Architecture & Core Components
The project follows a loose MVC (Model-View-Controller) pattern:
- **View (`app.py`)**: Defines the Gradio UI layout, event listeners, and state variables (`gr.State`). It delegates logic to the Controller.
- **Controller (`src/sam3_annotation_tool/controller.py`)**: Manages application state (`AppController`), including the current image, playlist, and annotations (`GlobalStore`). It coordinates between the UI and the Inference engine.
- **Model/Logic (`src/sam3_annotation_tool/inference.py`)**: Handles loading SAM3 models (`Sam3Model`, `Sam3TrackerModel`) and running inference.
- **Data Models (`src/sam3_annotation_tool/schemas.py`)**: Defines core data structures like `ObjectState`, `ProjectState`, and `SelectorInput`.
- **Helpers (`src/sam3_annotation_tool/view_helpers.py`)**: UI-specific utilities (drawing boxes on images, formatting dataframes).

## Development Workflow
- **Package Manager**: This project uses **`uv`** with `pyproject.toml` and `uv.lock`.
- **Syncing Deps**: Use `uv sync` to install dependencies (ensures exact versions).
- **Running the App**: Use `uv run app.py` to start the application.
- **Running Tests**: Use `uv run python -m pytest` or `uv run tests/test_ui_callbacks.py`.
- **Terminal Safety**: **ALWAYS** check the current working directory (`pwd`) before running terminal commands to ensure you are in the project root (`sam3-annotation-tool`).

## Critical Constraints & "Gotchas"
- **Gradio Version**: Must use `gradio==6.0.2` (or strictly pinned version in `pyproject.toml`). Newer unstable/nightly versions of Gradio cause the frontend to freeze (MutationObserver loops). Do NOT upgrade Gradio blindly.
- **License Separation**:
    - Code is **Apache 2.0**.
    - Model (SAM 3) is **Meta Custom License**.
    - **NEVER** bundle `.pt` or model weight files in the repo. They must be ignored in `.gitignore`.
- **Project Structure Change**: We migrated from `requirements.txt` to `pyproject.toml` + `uv.lock`. Do not use pip commands directly; use `uv add` or `uv sync`.

## Communication Protocol
- **Explain First**: Before making significant changes (especially to `src/` or core logic), explain the plan and reasoning.
- **Bash Suggestions**: Always provide the specific bash commands you intend to run or suggest the user runs.
- **Feedback Loop**: Don't perform long chains of silent edits. Stop and report progress.

## Coding Conventions
- **Gradio State**: Use `gr.State` for transient UI state (like current selection indices). Use `AppController` for persistent data (like annotations).
- **Error Handling**: In `app.py` callbacks, raise `gr.Error("Message")` to show user-friendly error notifications in the UI.
- **Path Handling**: Use `os.path.join` for cross-platform compatibility.
- **Imports**: Use absolute imports from `src` (e.g., `from src.sam3_annotation_tool.controller import controller`).

## Key Files
- `app.py`: Main entry point and UI definition.
- `src/sam3_annotation_tool/controller.py`: Central logic and state management.
- `src/sam3_annotation_tool/inference.py`: SAM3 model integration.
- `src/sam3_annotation_tool/schemas.py`: Data structures.
