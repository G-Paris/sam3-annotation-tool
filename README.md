# SAM3 Image Annotator

This project provides a Gradio interface for image annotation using the SAM3 model.

## Quickstart

### Prerequisites

1.  **Install `uv`**
    This project uses `uv` for extremely fast Python package management.
    
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
    2.  Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens).
    3.  Create a new token with **Read** permissions.
    4.  (Optional but recommended) Visit the [facebook/sam3 model card](https://huggingface.co/facebook/sam3) to accept any license terms if prompted.

### Installation & Running

1.  **Clone the repository**
    ```bash
    git clone <repository_url>
    cd SAM3_image_annotator
    ```

2.  **Set up the environment with `uv`**
    Create a virtual environment and install dependencies:
    ```bash
    # Create virtual environment
    uv venv

    # Activate the environment
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate

    # Install dependencies
    uv pip install -r requirements.txt
    ```

3.  **Run the Application**
    You need to provide your Hugging Face token when running the app. You can do this by setting the `HF_TOKEN` environment variable.

    **Linux/macOS:**
    ```bash
    export HF_TOKEN="hf_..."  # Replace with your actual token
    python app.py
    ```

    **Windows (PowerShell):**
    ```powershell
    $env:HF_TOKEN="hf_..."
    python app.py
    ```

    Alternatively, you can log in via the CLI if you have `huggingface_hub` installed (included in dependencies):
    ```bash
    huggingface-cli login
    # Follow the prompts to paste your token
    python app.py
    ```

4.  **Access the Interface**
    Once the application starts, open your browser and navigate to the local URL provided in the terminal (typically `http://127.0.0.1:7860`).

## Project Structure

- `app.py`: Main Gradio application entry point.
- `src/`: Source code for the application logic.
  - `controller.py`: Manages application state.
  - `inference.py`: Handles model loading and inference (SAM3).
  - `models.py`: Model definitions.
  - `theme.py`: Custom UI theme.
  - `utils.py`: Helper functions for image processing.
- `requirements.txt`: Python dependencies.
