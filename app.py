import gradio as gr
import numpy as np
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from src.sam3_annotation_tool.theme import CustomBlueTheme
from src.sam3_annotation_tool.controller import controller
from src.sam3_annotation_tool.inference import load_models
from src.sam3_annotation_tool.utils import apply_mask_overlay, get_bbox_from_mask
from src.sam3_annotation_tool.view_helpers import (
    draw_boxes_on_image, format_box_list, parse_dataframe, on_dataframe_change,
    delete_checked_boxes, on_upload, on_input_image_select, undo_last_click,
    on_crop_dataframe_change, format_crop_box
)

# Dynamically load example images from the example_img folder
EXAMPLE_IMAGES = sorted(glob.glob("example_img/*.jpg") + glob.glob("example_img/*.png") + glob.glob("example_img/*.jpeg"))

# Load models immediately on startup
load_models()

app_theme = CustomBlueTheme()

# --- Helper Functions ---
# (Moved to src/view_helpers.py)

def run_inference_step1(clean_image, text_prompt, boxes, labels, class_name_override, crop_box=None):
    """Step 1: Run Inference and switch screens."""
    print(f"🖱️  Run Inference Clicked! Prompt: '{text_prompt}', Override: '{class_name_override}', Boxes: {len(boxes)}, Crop: {crop_box}")
    
    if clean_image is None: 
        raise gr.Error("Please upload an image.")
    if not text_prompt and not boxes: 
        raise gr.Error("Please enter a text prompt or add box prompts.")
        
    # Only set image if not in playlist mode (to avoid resetting project state)
    if not controller.project.playlist:
        controller.set_image(clean_image)
    
    try:
        candidates = controller.search_and_add(text_prompt, boxes, labels, class_name_override, crop_box)
        print(f"✅ Search returned {len(candidates)} candidates.")
    except Exception as e:
        print(f"❌ Error during search: {e}")
        raise gr.Error(f"Inference failed: {str(e)}")
        
    # Return candidates, image, and screen visibility updates
    return (
        candidates,
        clean_image,
        gr.update(selected=2), # Switch to Results Tab
        gr.update(choices=controller.project.prompt_history),
        gr.update(choices=controller.project.class_name_history)
    )

def render_results_step2(candidates, image):
    """Step 2: Render List and Preview."""
    if image is None: return gr.update(), gr.update(), set()
    
    print("🖼️  Rendering results...")
    
    # Preview Image (All candidates dim)
    preview_img = controller.get_candidate_preview(candidates, selected_index=None)
        
    # List Items
    list_items = []
    for i, cand in enumerate(candidates):
        label = f"ID {i+1}: {cand.class_name} ({cand.score:.2f})"
        list_items.append(label)
        
    return (
        gr.update(choices=list_items, value=[]),
        gr.update(value=preview_img),
        set() # Reset selected indices
    )

def on_selection_change(selected_values, candidates):
    """Handle selection change in checkbox group."""
    # Reconstruct indices from selected values
    # Value format: "ID {i+1}: ..."
    selected_indices = set()
    
    # Create a map of label -> index for robust lookup
    label_map = {}
    for i, cand in enumerate(candidates):
        label = f"ID {i+1}: {cand.class_name} ({cand.score:.2f})"
        label_map[label] = i
        
    for val in selected_values:
        if val in label_map:
            selected_indices.add(label_map[val])
            
    # Update Preview Image
    # Pass the set of selected indices to highlight them
    preview_img = controller.get_candidate_preview(candidates, selected_index=selected_indices)
    
    return gr.update(value=preview_img), selected_indices

def select_all_candidates(candidates):
    """Select all candidates."""
    if not candidates: return gr.update(), gr.update(), set()
    
    all_indices = set(range(len(candidates)))
    
    list_items = []
    for i, cand in enumerate(candidates):
        label = f"ID {i+1}: {cand.class_name} ({cand.score:.2f})"
        list_items.append(label)
        
    # Preview Image - Highlight ALL
    preview_img = controller.get_candidate_preview(candidates, selected_index=all_indices)
        
    return gr.update(value=preview_img), gr.update(value=list_items), all_indices

def deselect_all_candidates(candidates):
    """Deselect all candidates."""
    if not candidates: return gr.update(), gr.update(), set()
    
    # Preview Image
    preview_img = controller.get_candidate_preview(candidates, selected_index=None)
        
    return gr.update(value=preview_img), gr.update(value=[]), set()

def add_to_store_wrapper(candidates, selected_indices):
    if not selected_indices: raise gr.Error("No masks selected.")
    # Convert set to list
    return add_to_store(candidates, list(selected_indices))

def toggle_click_mode(current_mode):
    """Toggle between Include and Exclude."""
    if "Include" in current_mode:
        return "Exclude (Red)"
    return "Include (Green)"

def revert_object_refinement(obj_id):
    """Revert object to initial state."""
    if not obj_id: return gr.update()
    controller.revert_object(obj_id)
    return init_editor(obj_id)[0]

def export_results(output_path, export_type="YOLO", zip_output=False):
    """Export results to output folder."""
    print(f"📦  Exporting results to {output_path} (Type: {export_type}, Zip: {zip_output})...")
    
    if "Not supported yet" in export_type:
        raise gr.Error(f"Export type '{export_type}' is not supported yet.")
        
    try:
        res = controller.export_data(output_path, purge=True, zip_output=zip_output)
        if res:
            _, msg = res
            return msg
        else:
            return "Export failed: No data to export."
    except Exception as e:
        return f"Export failed: {e}"

def add_to_store(candidates, selected_indices):
    if not selected_indices: raise gr.Error("No masks selected.")
    
    controller.add_candidates_to_store(candidates, selected_indices)
    
    return "Added to Store!", gr.update(selected=3) # Go to Editor Tab

# --- UI Layout ---

custom_css="""
#col-container { margin: 0 auto; max-width: 1400px; }
#main-title h1 { font-size: 2.1em !important; }
#input_image { position: relative; overflow: hidden; }
#input_image button, #input_image img, #input_image canvas { cursor: crosshair !important; }
.zoom-image img { transition: transform 0.1s ease-out; }

/* Use max-height to constrain large images without upscaling small ones */
/* This fixes zoom/click coordinate issues for images smaller than container */
#input_image img,
.zoom-image img {
    max-height: 600px !important;
    width: auto !important;
    height: auto !important;
}

/* Dataframe Font Size */
.box-list-df td, .box-list-df th, .box-list-df td span, .box-list-df td input, .box-list-df td div { font-size: 10px !important; line-height: 1.0 !important; padding: 2px !important; }
/* Hide Checkbox in Header for 'Del' column (assuming it's the first column) */
thead th:first-child input[type="checkbox"] { display: none !important; }

/* Column Widths */
.box-list-df th:nth-child(1), .box-list-df td:nth-child(1) { width: 30px !important; min-width: 30px !important; }
.box-list-df th:nth-child(2), .box-list-df td:nth-child(2) { width: 80px !important; }

/* Export Status Font Size */
#export-status textarea { font-size: 0.8em !important; }

/* Horizontal Radio Buttons */
.horizontal-radio .wrap { display: flex !important; flex-direction: row !important; gap: 10px !important; }
.horizontal-radio label { margin-bottom: 0 !important; align-items: center !important; }
.horizontal-radio span { font-size: 0.8em !important; }

/* Scrollable Radio List */
.scrollable-radio { max-height: 200px !important; overflow-y: auto !important; border: 1px solid #e5e7eb; padding: 5px; border-radius: 5px; }

/* Hide Footer */
footer { display: none !important; }

/* Ensure the tab bar is a flex container and takes full width */
.main-tabs > div:first-of-type {
    display: flex !important;
    width: 100% !important;
}

/* Navigation Row Compactness */
.nav-row {
    align-items: center !important;
    gap: 5px !important;
    margin-bottom: 5px !important;
}
.nav-status textarea {
    font-size: 0.8em !important;
    text-align: center !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 5px !important;
}
"""

# JS for Crosshair and Zoom
custom_js = """
function setupInteractions() {
    // Crosshair Logic
    const setupCrosshair = () => {
        const c = document.querySelector('#input_image');
        if (c && !c.dataset.crosshairSetup) {
            c.dataset.crosshairSetup = "true";
            c.style.position = 'relative';

            const createLine = (id, isH) => {
                let l = document.createElement('div');
                l.style.cssText = `position:absolute;background:cyan;pointer-events:none;z-index:10000;display:none;box-shadow:0 0 2px rgba(0,0,0,0.5);${isH ? 'height:1px;width:100%;' : 'width:1px;height:100%;top:0;'}`;
                c.appendChild(l);
                return l;
            };
            const h = createLine('h', true), v = createLine('v', false);

            c.addEventListener('mousemove', (e) => {
                const r = c.getBoundingClientRect();
                const x = e.clientX - r.left, y = e.clientY - r.top;
                if (x >= 0 && x <= r.width && y >= 0 && y <= r.height) {
                    h.style.display = v.style.display = 'block';
                    h.style.top = (y - 2) + 'px';
                    v.style.left = (x - 2) + 'px';
                } else { h.style.display = v.style.display = 'none'; }
            });
            c.addEventListener('mouseleave', () => { h.style.display = v.style.display = 'none'; });
        }
    };

    // Zoom Logic
    const setupZoom = () => {
        document.querySelectorAll('.zoom-image').forEach(container => {
            if (container.dataset.zoomSetup) return;
            container.dataset.zoomSetup = "true";
            container.style.overflow = 'hidden';
            
            let scale = 1, pointX = 0, pointY = 0, startX = 0, startY = 0, isDragging = false;

            container.addEventListener('wheel', (e) => {
                e.preventDefault();
                const img = container.querySelector('img');
                if (!img) return;
                
                img.style.transformOrigin = "0 0";
                img.style.transition = "transform 0.1s ease-out";

                const rect = container.getBoundingClientRect();
                const xs = (e.clientX - rect.left - pointX) / scale;
                const ys = (e.clientY - rect.top - pointY) / scale;
                
                const delta = -e.deltaY;
                (delta > 0) ? (scale *= 1.2) : (scale /= 1.2);
                if (scale < 1) scale = 1;

                pointX = e.clientX - rect.left - xs * scale;
                pointY = e.clientY - rect.top - ys * scale;

                img.style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
            });
            
            // Panning
            container.addEventListener('mousedown', (e) => {
                isDragging = true;
                startX = e.clientX - pointX;
                startY = e.clientY - pointY;
            });
            
            window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                e.preventDefault();
                const img = container.querySelector('img');
                if (!img) return;
                
                pointX = e.clientX - startX;
                pointY = e.clientY - startY;
                img.style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
            });

            window.addEventListener('mouseup', () => { isDragging = false; });
        });
    };

    // Observer
    const observer = new MutationObserver(() => {
        setupCrosshair();
        setupZoom();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    
    setupCrosshair();
    setupZoom();
}
"""

with gr.Blocks() as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    
    # State Variables
    st_boxes = gr.State([])
    st_labels = gr.State([])
    st_candidates = gr.State([])
    st_selected_indices = gr.State(set()) # Track selected indices
    st_current_image = gr.State(None)
    st_selected_box_index = gr.State(None) # Track selected box for deletion
    
    st_clean_input_image = gr.State(None) # Store original uploaded image
    st_pending_point = gr.State(None) # Store first point of box click
    st_crop_box = gr.State(None) # Store crop box [x1, y1, x2, y2]
    
    # Hidden status box for messages
    status_box = gr.Textbox(visible=False)
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **GP-engineering's SAM3 Annotator**", elem_id="main-title")
        
        with gr.Tabs(elem_classes=["main-tabs"]) as tabs:
            # --- SCREEN 0: SETUP ---
            with gr.TabItem("Setup", id=0) as setup_screen:
                with gr.Tabs():
                    with gr.TabItem("New Project"):
                        gr.Markdown("### Select Data Source")
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### Batch (Folder)")
                                upload_files = gr.File(label="Upload Folder", file_count="directory", file_types=["image"], height=200)
                            
                            with gr.Column():
                                gr.Markdown("#### Single Image")
                                single_image_input = gr.Image(
                                    label="Upload or Capture Image", 
                                    sources=["upload", "webcam", "clipboard"], 
                                    type="filepath",
                                    height=400
                                )
                                if EXAMPLE_IMAGES:
                                    gr.Examples(
                                        examples=EXAMPLE_IMAGES,
                                        inputs=single_image_input
                                    )
                        
                        gr.Markdown("### Project Details")
                        new_project_name = gr.Textbox(label="Project Name", placeholder="Enter a name for your new project")
                        start_btn = gr.Button("Start Annotation", variant="primary", interactive=False)
                    
                    with gr.TabItem("Save / Load"):
                        gr.Markdown("### Project State")
                        with gr.Row():
                            with gr.Column(scale=1):
                                project_name_input = gr.Textbox(label="Project Name", placeholder="my_project", value="my_project")
                                save_project_btn = gr.Button("Save Project State")
                            
                            with gr.Column(scale=1):
                                # Changed from File Upload to Dropdown for Local Projects
                                project_dropdown = gr.Dropdown(label="Load Existing Project", choices=[], interactive=True)
                                refresh_projects_btn = gr.Button("🔄 Refresh List", size="sm")
                                load_project_btn = gr.Button("Load Project State")
                                
                                with gr.Row():
                                    delete_project_btn = gr.Button("Delete Project", variant="stop", size="sm")
                                    confirm_delete_btn = gr.Button("⚠️ Confirm Delete", variant="stop", size="sm", visible=False)
                                    cancel_delete_btn = gr.Button("Cancel", size="sm", visible=False)
                                
                                with gr.Row():
                                    delete_all_btn = gr.Button("Delete ALL Projects", variant="stop", size="sm")
                                    confirm_delete_all_btn = gr.Button("⚠️ Confirm Delete ALL", variant="stop", size="sm", visible=False)
                                    cancel_delete_all_btn = gr.Button("Cancel", size="sm", visible=False)
                                
                            project_status = gr.Textbox(label="Status", interactive=False, lines=10, scale=1)

            # --- SCREEN 1: INPUT ---
            with gr.TabItem("Input", id=1) as input_screen:
                gr.Markdown("### Generate initial objects")
                
                # Navigation (Full Width)
                with gr.Row(elem_classes="nav-row"):
                    prev_btn = gr.Button("Previous", size="sm", scale=0)
                    nav_status = gr.Textbox(label="Status", show_label=False, value="0/0", interactive=False, scale=1, elem_classes="nav-status")
                    next_btn = gr.Button("Next", size="sm", scale=0)

                # Prompt Row (Full Width)
                with gr.Row():
                    txt_prompt = gr.Dropdown(label="Text Prompt", choices=[], allow_custom_value=True, show_label=True, scale=4)
                    txt_class_name = gr.Dropdown(label="Class Name Override", choices=[], allow_custom_value=True, show_label=True, scale=2)
                    run_btn = gr.Button("Run Inference", variant="primary", scale=1)

                with gr.Row():
                    # Left Column: Image
                    with gr.Column(scale=3):
                        img_input = gr.Image(
                            label="Current Image (Click 2 Points for Box)", 
                            type="pil", 
                            interactive=True,
                            elem_id="input_image",
                            elem_classes="zoom-image"
                        )
                    
                    # Right Column: Controls
                    with gr.Column(scale=1):
                        # Box Controls (Top Right)
                        with gr.Group():
                            # gr.Markdown("### Box Controls") # Removed header
                            click_effect = gr.Radio(["Crop Initial Image", "Include Area", "Exclude Area"], value="Include Area", label="Click Effect")
                            undo_click_btn = gr.Button("Undo Last Click", variant="secondary", size="sm")
                        
                        # Crop List
                        crop_list_display = gr.Dataframe(
                            headers=["Del", "x1", "y1", "x2", "y2"],
                            datatype=["bool", "number", "number", "number", "number"],
                            column_count=5,
                            interactive=True,
                            label="Crop Area (Model runs at 1024x1024)",
                            wrap=True,
                            elem_classes="box-list-df"
                        )

                        # Box List (Moved here)
                        gr.Markdown("")
                        # [Delete?, Type, x1, y1, x2, y2]
                        box_list_display = gr.Dataframe(
                            headers=["Del", "Type", "x1", "y1", "x2", "y2"], 
                            datatype=["bool", "str", "number", "number", "number", "number"],
                            column_count=6,
                            interactive=True,
                            label="Added Boxes",
                            wrap=True,
                            elem_classes="box-list-df"
                        )
                        delete_box_btn = gr.Button("Delete Checked Boxes", variant="stop", size="sm")

            # --- SCREEN 2: RESULTS ---
            with gr.TabItem("Results", id=2) as result_screen:
                with gr.Row():
                    gr.Markdown("### Select relevant objects")
                    result_img_counter = gr.Markdown("Image 0/0", elem_id="result-img-counter")

                with gr.Row():
                    with gr.Column(scale=3):
                        # Preview Image with ALL masks
                        preview_image = gr.Image(
                            label="Selected Candidates Preview", 
                            type="pil", 
                            interactive=False,
                            elem_classes="zoom-image"
                        )
                        
                    with gr.Column(scale=1):
                        # Checkbox List of candidates
                        results_list = gr.CheckboxGroup(
                            label="Candidates",
                            choices=[],
                            value=[],
                            interactive=True,
                            elem_classes="candidate-list"
                        )
                        
                        with gr.Row():
                            select_all_btn = gr.Button("Select All", size="sm", variant="secondary")
                            deselect_all_btn = gr.Button("Deselect All", size="sm", variant="secondary")
                            
                        with gr.Row():
                            confirm_btn = gr.Button("Add Selected to Store", variant="primary")
                        
                        with gr.Row():
                            add_more_btn = gr.Button("Add Selected & Generate More", variant="secondary")

            # --- SCREEN 3: EDITOR ---
            with gr.TabItem("Editor", id=3) as editor_screen:
                with gr.Row():
                    gr.Markdown("### Refine individual objects")
                    editor_img_counter = gr.Markdown("Image 0/0", elem_id="editor-img-counter")
                    refresh_editor_btn = gr.Button("🔄 Refresh", size="sm", scale=0)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Main interactive image for refinement
                        refine_image = gr.Image(
                            label="Click to Refine",
                            type="pil",
                            interactive=False,
                            elem_classes="zoom-image"
                        )
                        
                        # Export moved to separate tab

                    with gr.Column(scale=1):
                        gr.Markdown("")
                        
                        with gr.Row():
                            object_list = gr.Radio(
                                label="Select Object",
                                choices=[],
                                interactive=True,
                                elem_classes="scrollable-radio"
                            )
                        
                        with gr.Row():
                            # revert_btn = gr.Button("Revert", size="sm", variant="secondary") # Moved below
                            delete_btn = gr.Button("Delete", size="sm", variant="stop")      
                        
                        gr.Markdown("")
                        with gr.Row():
                            click_mode = gr.Radio(["Include (Green)", "Exclude (Red)"], value="Include (Green)", label="Click Mode", interactive=True, elem_classes="horizontal-radio", scale=2)
                            undo_btn = gr.Button("Undo Last Click", variant="secondary", size="sm", scale=1)
                        
                        revert_btn = gr.Button("Revert Object", size="sm", variant="secondary")

                        gr.Markdown("")
                        with gr.Row():
                            finish_img_btn = gr.Button("Finish & Next Image", variant="primary")
                            finish_save_btn = gr.Button("Finish (Save)", variant="secondary")

            # --- SCREEN 4: EXPORT ---
            with gr.TabItem("Export", id=4) as export_screen:              
                gr.Markdown("### Export data")
                with gr.Row():
                    with gr.Column():
                        # Project State Display
                        export_status_display = gr.JSON(label="Ready for Export", value={})
                        
                        with gr.Row():
                            txt_output_dir = gr.Textbox(label="Output Folder", value="output", scale=3)
                            export_type = gr.Dropdown(label="Export Type", choices=["YOLO", "COCO (Not supported yet)"], value="YOLO", scale=1)
                        
                        gr.Markdown("⚠️ **Warning:** Exporting will delete the current contents in the output directory before saving.")
                        
                        with gr.Row():
                            zip_export = gr.Checkbox(label="Zip Output", value=False, scale=0)
                            export_btn = gr.Button("Export", scale=0, min_width=150)
                            reset_btn = gr.Button("Reset Project", variant="stop", scale=0, min_width=150)
                        
                        gr.Markdown("ℹ️ **Note:** All classes in the project history will be included in `data.yaml` to ensure compatibility with tools like CVAT, make sure all labels in the CVAT project are added to the project history (prevents KeyError).")
                            
                        export_status = gr.Textbox(label="Export Status", interactive=False, elem_id="export-status", lines=5)

    # --- Helper Functions for Editor ---
    
    def init_editor(selected_obj_id=None):
        """Initialize editor screen with current image and objects."""
        base_img = controller.current_image
        if base_img is None: return None, gr.update(choices=[])
        
        # Create choices for Radio
        choices = []
        for obj_id, obj in controller.store.objects.items():
            # Limit ID display to first 4 chars
            display_id = obj_id[:4]
            choices.append((f"{obj.class_name} ({display_id})", obj_id))
            
        # Determine selection
        if selected_obj_id is None and choices:
            selected_obj_id = choices[0][1]
        elif selected_obj_id and selected_obj_id not in [c[1] for c in choices]:
             selected_obj_id = choices[0][1] if choices else None

        # Create overlay
        overlay_img = base_img.copy()
        
        draw = ImageDraw.Draw(overlay_img)
        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        if selected_obj_id and selected_obj_id in controller.store.objects:
            # Show ONLY selected object (as per original logic)
            
            obj = controller.store.objects[selected_obj_id]
            mask = obj.binary_mask
            overlay_img = apply_mask_overlay(base_img, np.array([mask]), opacity=0.6)
            
            # Draw Points
            draw = ImageDraw.Draw(overlay_img)
            radius = 5
            for pt, lbl in zip(obj.input_points, obj.input_labels):
                color = "#00FF00" if lbl == 1 else "#FF0000"
                x, y = pt
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white")
                
            # Draw ID
            bbox = get_bbox_from_mask(mask)
            if bbox:
                x, y = bbox[0], bbox[1]
                draw.text((x, y - 20), selected_obj_id[:5], fill="white", font=font, stroke_width=2, stroke_fill="black")
        
        return overlay_img, gr.update(choices=choices, value=selected_obj_id)

    def on_image_click(img, evt: gr.SelectData, obj_id, mode):
        """Handle click on image to refine object."""
        if not obj_id: raise gr.Error("Please select an object to refine first.")
        
        point = [evt.index[0], evt.index[1]]
        label = 1 if "Include" in mode else 0
        
        # Call controller
        controller.refine_object(obj_id, point, label)
        
        # Re-render overlay
        return init_editor(obj_id)[0]

    def on_undo(obj_id):
        if not obj_id: return gr.update()
        controller.undo_last_point(obj_id)
        return init_editor(obj_id)[0]

    def on_delete(obj_id):
        if not obj_id: return gr.update(), gr.update()
        controller.remove_object(obj_id)
        
        # Auto-save
        controller.auto_save()
        
        # Refresh everything
        img, radio = init_editor(None)
        return img, radio

    # --- Event Wiring ---
    
    # 1. Upload Files
    def handle_upload(files):
        # Load playlist
        img, _, _, _ = on_upload(files)
        # Enable start button if images found
        count = len(controller.project.playlist)
        if count > 0:
            return gr.update(interactive=True, value=f"Start Annotation ({count} images)")
        else:
            return gr.update(interactive=False, value="Start Annotation")

    upload_files.upload(
        fn=handle_upload,
        inputs=[upload_files],
        outputs=[start_btn]
    )
    
    def handle_single_image(file_path):
        if not file_path:
            return gr.update(interactive=False, value="Start Annotation")
        
        # Reuse on_upload logic which expects a list of paths
        img, _, _, _ = on_upload([file_path])
        
        if controller.project.playlist:
             return gr.update(interactive=True, value="Start Annotation (1 Image)")
        return gr.update(interactive=False, value="Start Annotation")

    single_image_input.change(
        fn=handle_single_image,
        inputs=[single_image_input],
        outputs=[start_btn]
    )
    
    # Project Save/Load
    def on_save_project(name):
        if not name: return "Please enter a project name."
        
        # Create saved_projects directory
        import os
        os.makedirs("saved_projects", exist_ok=True)
        
        filename = f"saved_projects/{name}.json"
        success, msg = controller.save_project(filename)
        return msg

    save_project_btn.click(
        fn=on_save_project,
        inputs=[project_name_input],
        outputs=[project_status]
    )

    def list_projects():
        import os
        if not os.path.exists("saved_projects"):
            return []
        files = [f for f in os.listdir("saved_projects") if f.endswith(".json")]
        return sorted(files)

    def on_load_project(filename):
        if not filename: return "No project selected.", gr.update(), gr.update(), None, [], [], None, "0/0", None, gr.update(), gr.update(), {}, gr.update(), gr.update()
        
        file_path = f"saved_projects/{filename}"
        success, msg = controller.load_project(file_path)
        
        if success:
            # Update UI state
            img = controller.current_image
            status = f"Image {controller.project.current_index + 1}/{len(controller.project.playlist)}"
            return (
                msg,
                gr.update(selected=1), # Go to Input
                gr.update(value=img, interactive=True),
                img, [], [], None, status, None, gr.update(value=[]), gr.update(value="Crop Initial Image"),
                get_export_status(),
                gr.update(choices=controller.project.prompt_history),
                gr.update(choices=controller.project.class_name_history)
            )
        return msg, gr.update(), gr.update(), None, [], [], None, "0/0", None, gr.update(), gr.update(), {}, gr.update(), gr.update()

    load_project_btn.click(
        fn=on_load_project,
        inputs=[project_dropdown],
        outputs=[project_status, tabs, img_input, st_clean_input_image, st_boxes, st_labels, st_pending_point, nav_status, st_crop_box, crop_list_display, click_effect, export_status_display, txt_prompt, txt_class_name]
    )
    
    refresh_projects_btn.click(
        fn=lambda: gr.update(choices=list_projects()),
        inputs=[],
        outputs=[project_dropdown]
    )
    
    # Delete Project Logic
    def on_delete_click(filename):
        if not filename: return gr.update(visible=False), gr.update(visible=False), "Please select a project to delete."
        return gr.update(visible=True), gr.update(visible=True), f"Are you sure you want to delete '{filename}'? This cannot be undone."

    delete_project_btn.click(
        fn=on_delete_click,
        inputs=[project_dropdown],
        outputs=[confirm_delete_btn, cancel_delete_btn, project_status]
    )

    def on_cancel_delete():
        return gr.update(visible=False), gr.update(visible=False), "Deletion cancelled."

    cancel_delete_btn.click(
        fn=on_cancel_delete,
        inputs=[],
        outputs=[confirm_delete_btn, cancel_delete_btn, project_status]
    )

    def on_confirm_delete(filename):
        if not filename: return gr.update(visible=False), gr.update(visible=False), "No project selected.", gr.update()
        
        import os
        import shutil
        
        json_path = f"saved_projects/{filename}"
        assets_dir = f"saved_projects/{filename.replace('.json', '')}_assets"
        
        msg = []
        try:
            if os.path.exists(json_path):
                os.remove(json_path)
                msg.append(f"Deleted {filename}")
            
            if os.path.exists(assets_dir):
                shutil.rmtree(assets_dir)
                msg.append(f"Deleted assets folder")
                
            if not msg:
                return gr.update(visible=False), gr.update(visible=False), "Project files not found.", gr.update(choices=list_projects())
                
            return gr.update(visible=False), gr.update(visible=False), f"Success: {', '.join(msg)}", gr.update(choices=list_projects(), value=None)
            
        except Exception as e:
            return gr.update(visible=False), gr.update(visible=False), f"Error deleting project: {e}", gr.update(choices=list_projects())

    confirm_delete_btn.click(
        fn=on_confirm_delete,
        inputs=[project_dropdown],
        outputs=[confirm_delete_btn, cancel_delete_btn, project_status, project_dropdown]
    )

    # Delete All Projects Logic
    def on_delete_all_click():
        return gr.update(visible=True), gr.update(visible=True), "Are you sure you want to delete ALL projects? This cannot be undone."

    delete_all_btn.click(
        fn=on_delete_all_click,
        inputs=[],
        outputs=[confirm_delete_all_btn, cancel_delete_all_btn, project_status]
    )

    def on_cancel_delete_all():
        return gr.update(visible=False), gr.update(visible=False), "Deletion cancelled."

    cancel_delete_all_btn.click(
        fn=on_cancel_delete_all,
        inputs=[],
        outputs=[confirm_delete_all_btn, cancel_delete_all_btn, project_status]
    )

    def on_confirm_delete_all():
        import os
        import shutil
        
        folder = "saved_projects"
        if not os.path.exists(folder):
             return gr.update(visible=False), gr.update(visible=False), "No projects found.", gr.update(choices=[])
             
        try:
            # Delete all files and folders in saved_projects
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            
            return gr.update(visible=False), gr.update(visible=False), "All projects deleted.", gr.update(choices=[], value=None)
            
        except Exception as e:
            return gr.update(visible=False), gr.update(visible=False), f"Error deleting projects: {e}", gr.update(choices=list_projects())

    confirm_delete_all_btn.click(
        fn=on_confirm_delete_all,
        inputs=[],
        outputs=[confirm_delete_all_btn, cancel_delete_all_btn, project_status, project_dropdown]
    )
    
    # Init project list on load
    demo.load(fn=lambda: gr.update(choices=list_projects()), inputs=[], outputs=[project_dropdown])
    
    def start_session(project_name):
        if not controller.project.playlist:
             raise gr.Error("No images loaded.")
        
        # Initialize project path only if name provided
        if project_name:
            import os
            os.makedirs("saved_projects", exist_ok=True)
            filename = f"saved_projects/{project_name}.json"
            
            # Check if exists
            if os.path.exists(filename):
                 # Optional: Warn user? For now, just proceed (will overwrite if saved)
                 pass
                 
            controller.active_project_path = filename
            
            # Initial Save
            controller.save_project(filename)
        else:
            controller.active_project_path = None
        
        # Ensure we have the first image loaded
        if controller.current_image is None:
            print("⚠️ Current image is None, attempting to load index 0...")
            controller.load_image_at_index(0)
            
        img = controller.current_image
        if img is None:
             raise gr.Error("Failed to load first image.")
             
        status = f"Image {controller.project.current_index + 1}/{len(controller.project.playlist)}"
        
        return (
            gr.update(selected=1), # Switch to Input Tab
            gr.update(value=img, interactive=True), # img_input - Explicit update
            img, # st_clean_input_image
            [], # st_boxes
            [], # st_labels
            None, # st_pending
            status, # nav_status
            None, # st_crop_box
            gr.update(value=[]), # crop_list_display
            gr.update(value="Crop Initial Image") # click_effect
        )

    start_btn.click(
        fn=start_session,
        inputs=[new_project_name],
        outputs=[tabs, img_input, st_clean_input_image, st_boxes, st_labels, st_pending_point, nav_status, st_crop_box, crop_list_display, click_effect]
    )
    
    # Navigation
    def on_nav_prev():
        img = controller.prev_image()
        status = f"Image {controller.project.current_index + 1}/{len(controller.project.playlist)}" if img else "0/0"
        return img, img, [], [], None, status

    def on_nav_next():
        img = controller.next_image()
        status = f"Image {controller.project.current_index + 1}/{len(controller.project.playlist)}" if img else "0/0"
        return img, img, [], [], None, status

    prev_btn.click(
        fn=on_nav_prev,
        outputs=[img_input, st_clean_input_image, st_boxes, st_labels, st_pending_point, nav_status]
    )
    
    next_btn.click(
        fn=on_nav_next,
        outputs=[img_input, st_clean_input_image, st_boxes, st_labels, st_pending_point, nav_status]
    )
    
    # 2. Click on Image (Add Box)
    img_input.select(
        fn=on_input_image_select,
        inputs=[st_pending_point, st_boxes, st_labels, click_effect, st_clean_input_image, st_crop_box],
        outputs=[img_input, st_pending_point, st_boxes, st_labels, box_list_display, st_crop_box, crop_list_display]
    )
    
    # 2b. Undo Click
    undo_click_btn.click(
        fn=undo_last_click,
        inputs=[st_pending_point, st_boxes, st_labels, st_clean_input_image, st_crop_box],
        outputs=[img_input, st_pending_point, st_boxes, st_labels, box_list_display, st_crop_box, crop_list_display]
    )
    
    # 3. Dataframe Edits
    box_list_display.change(
        fn=on_dataframe_change,
        inputs=[box_list_display, st_clean_input_image, st_crop_box],
        outputs=[img_input, st_boxes, st_labels]
    )

    crop_list_display.change(
        fn=on_crop_dataframe_change,
        inputs=[crop_list_display, st_clean_input_image, st_boxes, st_labels],
        outputs=[img_input, st_crop_box]
    )
    
    # 3b. Delete Checked
    delete_box_btn.click(
        fn=delete_checked_boxes,
        inputs=[box_list_display, st_clean_input_image, st_crop_box],
        outputs=[st_boxes, st_labels, box_list_display, img_input]
    )
    
    # 4. Run Inference (Button + Enter)
    run_inference_fn = lambda img, txt, boxes, labels, cls_name, crop: run_inference_step1(img, txt, boxes, labels, cls_name, crop)
    
    def start_inference(img, prompt, boxes):
        if img is None:
             raise gr.Error("Please upload an image.")
        if not prompt and not boxes:
             raise gr.Error("Please enter a text prompt or add box prompts.")
        return gr.update(value="Running Inference...", interactive=False)

    # Event listeners for inference are defined at the end of the file to include counter updates

    
    # 3b. Select All
    select_all_btn.click(
        fn=select_all_candidates,
        inputs=[st_candidates],
        outputs=[preview_image, results_list, st_selected_indices]
    )

    deselect_all_btn.click(
        fn=deselect_all_candidates,
        inputs=[st_candidates],
        outputs=[preview_image, results_list, st_selected_indices]
    )
    
    # 3c. List Select
    results_list.change(
        fn=on_selection_change,
        inputs=[results_list, st_candidates],
        outputs=[preview_image, st_selected_indices]
    )
    
    # 5. Confirm Selection -> Go to Editor
    confirm_btn.click(
        fn=add_to_store_wrapper,
        inputs=[st_candidates, st_selected_indices],
        outputs=[status_box, tabs]
    ).then(
        fn=init_editor,
        inputs=[],
        outputs=[refine_image, object_list]
    )
    
    # 6. Editor Interactions
    object_list.change(
        fn=init_editor,
        inputs=[object_list],
        outputs=[refine_image, object_list]
    )

    refine_image.select(
        fn=on_image_click,
        inputs=[refine_image, object_list, click_mode],
        outputs=[refine_image]
    )
    
    undo_btn.click(
        fn=on_undo,
        inputs=[object_list],
        outputs=[refine_image]
    )
    
    revert_btn.click(
        fn=revert_object_refinement,
        inputs=[object_list],
        outputs=[refine_image]
    )
    
    delete_btn.click(
        fn=on_delete,
        inputs=[object_list],
        outputs=[refine_image, object_list]
    )
    
    export_btn.click(
        fn=export_results,
        inputs=[txt_output_dir, export_type, zip_export],
        outputs=[export_status]
    )
    
    # Helper to get project status
    def get_project_status():
        if not controller.project: return {}
        
        # Build dict directly to avoid type inference issues
        details = {}
        for path, store in controller.project.annotations.items():
            name = path.split("/")[-1]
            details[name] = len(store.objects)

        stats = {
            "total_images": len(controller.project.playlist),
            "current_index": controller.project.current_index,
            "annotated_images": len(controller.project.annotations),
            "total_objects": sum(len(s.objects) for s in controller.project.annotations.values()),
            "details": details
        }
        
        return stats

    # Helper for Export Status
    def get_export_status():
        if not controller.project: return {}
        
        # Map paths to indices
        playlist_map = {path: i for i, path in enumerate(controller.project.playlist)}
        
        finished_images = []
        total_objects = 0
        
        # Sort by index
        sorted_annotations = sorted(
            controller.project.annotations.items(),
            key=lambda x: playlist_map.get(x[0], -1)
        )
        
        for path, store in sorted_annotations:
            idx = playlist_map.get(path, -1)
            count = len(store.objects)
            
            # Only include images that have objects
            if count == 0:
                continue

            # Include if it has objects or is in the annotations map (meaning visited/saved)
            total_objects += count
            name = path.split("/")[-1]
            finished_images.append({
                "index": idx + 1, # 1-based index for display
                "filename": name,
                "object_count": count
            })
                
        return {
            "total_objects_annotated": total_objects,
            "finished_images_count": len(finished_images),
            "finished_images_list": finished_images
        }

    # Helper to get current image counter string
    def get_image_counter():
        if not controller.project.playlist: return "Image 0/0"
        return f"Image {controller.project.current_index + 1}/{len(controller.project.playlist)}"

    # Finish Image & Next
    def on_finish_image():
        print("🏁 Finishing image...")
        # Ensure current state is saved before moving
        if controller.current_image_path:
            controller.project.annotations[controller.current_image_path] = controller.store
            
        # Auto-save project
        controller.auto_save()
            
        img = controller.next_image()
        status = get_image_counter() if img else "Finished"
        
        if img:
            print(f"➡️ Next image loaded: {controller.project.current_index}")
            return (
                gr.update(selected=1), # Switch to Input Tab
                gr.update(value=img, interactive=True), # img_input
                img, # st_clean_input_image
                [], # st_boxes
                [], # st_labels
                None, # st_pending
                status, # nav_status
                None, # st_crop_box
                gr.update(value=[]),  # crop_list_display
                gr.update(value="Crop Initial Image"), # click_effect
                get_export_status() # Update export status
            )
        else:
            print("🛑 Playlist finished.")
            # End of playlist
            return (
                gr.update(selected=3), # Stay on Editor Tab
                gr.update(), 
                gr.update(),
                [], [], None, 
                "Finished",
                gr.update(),
                gr.update(),
                gr.update(), # click_effect
                get_export_status() # Update export status
            )

    finish_img_btn.click(
        fn=on_finish_image,
        outputs=[tabs, img_input, st_clean_input_image, st_boxes, st_labels, st_pending_point, nav_status, st_crop_box, crop_list_display, click_effect, export_status_display]
    )
    
    # Finish (Save Only) Logic
    def on_finish_save():
        print("💾 Saving current image state...")
        # Save current state
        if controller.current_image_path:
            controller.project.annotations[controller.current_image_path] = controller.store
            
        # Auto-save project
        controller.auto_save()
        
        # Update status display (but don't move tabs)
        return get_export_status()

    finish_save_btn.click(
        fn=on_finish_save,
        outputs=[export_status_display]
    )
    
    # Add & Generate More Logic
    def add_and_restart(candidates, selected_indices, clean_img):
        if not selected_indices: raise gr.Error("No masks selected.")
        controller.add_candidates_to_store(candidates, selected_indices)
        
        # Auto-save
        controller.auto_save()
        
        # Return updates to switch to Input tab and clear prompts
        return (
            gr.update(selected=1), # Switch to Input
            gr.update(value=clean_img, interactive=True), # Reset img_input
            [], # st_boxes (clear)
            [], # st_labels (clear)
            None, # st_pending (clear)
            gr.update(value=[]), # box_list_display (clear)
            gr.update(value="Crop Initial Image") # Reset click effect
        )

    add_more_btn.click(
        fn=add_and_restart,
        inputs=[st_candidates, st_selected_indices, st_clean_input_image],
        outputs=[tabs, img_input, st_boxes, st_labels, st_pending_point, box_list_display, click_effect]
    )
    
    # Update status on enter editor
    confirm_btn.click(
        fn=add_to_store_wrapper,
        inputs=[st_candidates, st_selected_indices],
        outputs=[status_box, tabs]
    ).then(
        fn=lambda: (controller.auto_save(), None)[1], # Auto-save on confirm
        inputs=[],
        outputs=[]
    ).then(
        fn=init_editor,
        inputs=[],
        outputs=[refine_image, object_list]
    ).then(
        fn=get_image_counter,
        outputs=[editor_img_counter]
    )
    
    # Update counter on enter results
    run_btn.click(
        fn=start_inference,
        inputs=[st_clean_input_image, txt_prompt, st_boxes],
        outputs=[run_btn]
    ).then(
        fn=run_inference_fn,
        inputs=[st_clean_input_image, txt_prompt, st_boxes, st_labels, txt_class_name, st_crop_box],
        outputs=[st_candidates, st_current_image, tabs, txt_prompt, txt_class_name]
    ).then(
        fn=render_results_step2,
        inputs=[st_candidates, st_current_image],
        outputs=[results_list, preview_image, st_selected_indices]
    ).then(
        fn=get_image_counter,
        outputs=[result_img_counter]
    ).then(
        fn=lambda: gr.update(value="Run Inference", interactive=True),
        inputs=[],
        outputs=[run_btn]
    )
    
    # Reset Project Logic
    def on_reset():
        controller.reset_project()
        return (
            gr.update(selected=0), # Go to Setup
            gr.update(value={}), # Clear export status display
            gr.update(value=""), # Clear export log
            gr.update(interactive=False, value="Start Annotation"), # Reset start button
            None, # Clear current image
            [], [], None, # Clear boxes, labels, pending
            "0/0", # Clear nav status
            None, # Clear crop box
            gr.update(value=[]), # Clear crop list
            gr.update(value="Crop Initial Image"), # Reset click effect
            gr.update(value="", choices=[]), # Clear prompt history
            gr.update(value="", choices=[])  # Clear class name history
        )

    reset_btn.click(
        fn=on_reset,
        inputs=[],
        outputs=[tabs, export_status_display, export_status, start_btn, img_input, st_boxes, st_labels, st_pending_point, nav_status, st_crop_box, crop_list_display, click_effect, txt_prompt, txt_class_name]
    )
    
    # Tab Switching Logic
    def on_tab_select(evt: gr.SelectData):
        print(f"Tab selected: {evt.value}, Index: {evt.index}")
        if evt.value == "Editor" or evt.index == 3:
            img, radio = init_editor(None)
            counter = get_image_counter()
            print(f"Editor Init: Image={img is not None}, Counter={counter}")
            return img, radio, counter
        return gr.update(), gr.update(), gr.update()

    tabs.select(
        fn=on_tab_select,
        inputs=[],
        outputs=[refine_image, object_list, editor_img_counter]
    )
    
    # Refresh Button Logic
    def on_refresh_editor():
        img, radio = init_editor(None)
        counter = get_image_counter()
        return img, radio, counter

    refresh_editor_btn.click(
        fn=on_refresh_editor,
        inputs=[],
        outputs=[refine_image, object_list, editor_img_counter]
    )

    # Load JS
    demo.load(None, None, None, js=custom_js)

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=app_theme, ssr_mode=False, mcp_server=False, show_error=True)
