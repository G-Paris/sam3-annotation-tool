# SAM3 Annotation Tool

GP's interactive Gradio app to generate train datasets with annotations for one or multiple images using SAM3.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/gp187/sam3-annotation-tool)

https://github.com/user-attachments/assets/b34d1784-6013-45ee-99e8-c99be51cabe3

This was a big exploration of how to vibe-code my way through such a project. So give the tool some slack if you're trying to make some changes to it :)

## Features

*   **Generate masks using prompts, input boxes or a combination**
*   **Discard masks that are not relevant**
*   **Finetune masks using point refinement**
*   **Export dataset zipped or unzipped, zipped uploads directly to cvat**

## Extra
*   **Zoom and move around higher res images**
*   **Define cropped area to increase SAM3 inference resolution**
*   **Append masks from multiple SAM inference calls per image**
*   **Save&load progress when working with larger amount of images**

## Quickstart

### Prerequisites

1.  **Install `uv`**
    This project uses `uv` for fast Python package management.
    
    **Linux/macOS:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    
    **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Hugging Face Token**
    The SAM3 model (`facebook/sam3`) is hosted on Hugging Face and may be gated. You need an access token to download it.
    
    1.  Create a Hugging Face account if you don't have one.
    2.  **Request Access**: Visit the [facebook/sam3 model card](https://huggingface.co/facebook/sam3) and accept the license terms to request access. You must do this before the model can be downloaded.
    3.  Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens).
    4.  Create a new token with **Read** permissions. You only need read access to the SAM3 repository.

### Installation & Running

1.  **Clone the repository**
    ```bash
    git clone https://github.com/G-Paris/sam3-annotation-tool.git
    cd sam3-annotation-tool
    ```

2.  **Install dependencies**
    Use `uv` to sync the project environment:
    ```bash
    # Create virtual environment and install dependencies from uv.lock
    uv sync
    ```

3.  **Run the Application**
    Use `uv run` to start the app.

    You must provide your Hugging Face token (with access to `facebook/sam3`).

    **Linux/macOS:**
    ```bash
    export HF_TOKEN="hf_..." 
    uv run app.py
    ```

    **Windows (PowerShell):**
    ```powershell
    $env:HF_TOKEN="hf_..."
    uv run app.py
    ```

    *Tip: You can also create a `.env` file (not committed) to store your token.*

4.  **Access the Interface**
    Once the application starts, open your browser and navigate to the local URL provided in the terminal (typically `http://127.0.0.1:7860`).

## License & Legal Disclaimer

**This Tool (The Code): Apache License 2.0**
The source code of this annotation tool (UI, wrapper logic, data management) is licensed under the [Apache License 2.0](LICENSE). You are free to use, modify, and distribute this software for commercial and non-commercial purposes, provided you include the original license.

**The Model (SAM 3): Meta Custom License**
This tool uses the **Segment Anything Model 3 (SAM 3)** by Meta. The model weights and core SAM 3 definitions are subject to **Meta's Custom License**, which may have different restrictions (especially regarding commercial use or redistributing the weights).

*   **We do NOT bundle SAM 3 weights** in this repository.
*   The model is downloaded automatically from Hugging Face (`facebook/sam3`) using your access token.
*   By using this tool, you agree to comply with Meta's license terms for SAM 3.

**Please review the separate license terms for SAM 3 explicitly on their [Hugging Face Model Card](https://huggingface.co/facebook/sam3).**

## Project Structure

- `app.py`: Main Gradio application entry point and UI definition.
- `src/sam3_annotation_tool/`: Source code for the application logic.
  - `controller.py`: Manages application state and business logic.
  - `inference.py`: Handles model loading and inference (SAM3).
  - `schemas.py`: Pydantic data models for type safety.
  - `view_helpers.py`: UI-specific helper functions and event handlers.
  - `dataset_manager.py`: YOLO dataset export and management.
  - `theme.py`: Custom UI theme.
  - `utils.py`: General helper functions for image processing.
- `scripts/`: Utility scripts for dataset operations (cleanup, verification, analysis).
- `tests/`: Unit and integration tests.
- `pyproject.toml`: Project configuration and dependencies.
- `uv.lock`: Locked dependency versions for reproducibility.
