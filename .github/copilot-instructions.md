# SAM3 Image Annotator - Copilot Instructions

## Project Overview
This is a Python-based image annotation tool using the **SAM3 (Segment Anything Model 3)** model, built with a **Gradio** web interface.

## Architecture & Core Components
The project follows a loose MVC (Model-View-Controller) pattern:
- **View (`app.py`)**: Defines the Gradio UI layout, event listeners, and state variables (`gr.State`). It delegates logic to the Controller.
- **Controller (`src/controller.py`)**: Manages application state (`AppController`), including the current image, playlist, and annotations (`GlobalStore`). It coordinates between the UI and the Inference engine.
- **Model/Logic (`src/inference.py`)**: Handles loading SAM3 models (`Sam3Model`, `Sam3TrackerModel`) and running inference.
- **Data Models (`src/schemas.py`)**: Defines core data structures like `ObjectState`, `ProjectState`, and `SelectorInput`.
- **Helpers (`src/view_helpers.py`)**: UI-specific utilities (drawing boxes on images, formatting dataframes).

## Development Workflow
- **Package Manager**: This project uses **`uv`** for dependency management and execution.
- **Running the App**: Use `uv run app.py` to start the application.
- **Running Tests**: Use `uv run python -m pytest` or `uv run tests/test_ui_callbacks.py`.
- **Terminal Safety**: **ALWAYS** check the current working directory (`pwd`) before running terminal commands to ensure you are in the project root (`SAM3_image_annotator`).

## Communication Protocol
- **Explain First**: Before making significant changes (especially to `src/` or core logic), explain the plan and reasoning.
- **Bash Suggestions**: Always provide the specific bash commands you intend to run or suggest the user runs.
- **Feedback Loop**: Don't perform long chains of silent edits. Stop and report progress.

## Coding Conventions
- **Gradio State**: Use `gr.State` for transient UI state (like current selection indices). Use `AppController` for persistent data (like annotations).
- **Error Handling**: In `app.py` callbacks, raise `gr.Error("Message")` to show user-friendly error notifications in the UI.
- **Path Handling**: Use `os.path.join` for cross-platform compatibility.
- **Imports**: Use absolute imports from `src` (e.g., `from src.controller import controller`).

## Key Files
- `app.py`: Main entry point and UI definition.
- `src/controller.py`: Central logic and state management.
- `src/inference.py`: SAM3 model integration.
- `src/schemas.py`: Data structures.
