import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.theme import CustomBlueTheme
from src.controller import controller
from src.inference import load_models
from src.utils import apply_mask_overlay, draw_points_on_image, get_bbox_from_mask, create_mask_crop
from src.view_helpers import (
    draw_boxes_on_image, format_box_list, parse_dataframe, on_dataframe_change,
    delete_checked_boxes, on_upload, on_input_image_select, undo_last_click,
    on_crop_dataframe_change, format_crop_box
)

# Load models immediately on startup
load_models()

app_theme = CustomBlueTheme()

# --- Helper Functions ---
# (Moved to src/view_helpers.py)

def run_inference_step1(clean_image, text_prompt, boxes, labels, class_name_override, crop_box=None):
    """Step 1: Run Inference and switch screens."""
    print(f"🖱️ Run Inference Clicked! Prompt: '{text_prompt}', Override: '{class_name_override}', Boxes: {len(boxes)}, Crop: {crop_box}")
    
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
        gr.update(selected=2) # Switch to Results Tab
    )

def render_results_step2(candidates, image):
    """Step 2: Render Gallery and Preview."""
    if image is None: return gr.update(), gr.update(), set()
    
    print("🖼️ Rendering results...")
    
    # Preview Image (All candidates dim)
    preview_img = image.copy()
    if candidates:
        all_masks = np.array([c.binary_mask for c in candidates])
        preview_img = apply_mask_overlay(preview_img, all_masks, opacity=0.5)
        
    # Gallery Items
    gallery_items = []
    for i, cand in enumerate(candidates):
        crop = create_mask_crop(image, cand.binary_mask)
        label = f"{cand.class_name} ({cand.score:.2f})"
        gallery_items.append((crop, label))
        
    return (
        gr.update(value=gallery_items),
        gr.update(value=preview_img),
        set() # Reset selected indices
    )

def on_gallery_select(evt: gr.SelectData, selected_indices, candidates):
    """Handle selection in gallery."""
    idx = evt.index
    if idx is None: return gr.update(), gr.update(), selected_indices
    
    # Toggle selection
    if idx in selected_indices:
        selected_indices.remove(idx)
    else:
        selected_indices.add(idx)
        
    # Update Preview Image
    base_img = controller.current_image
    if base_img is None: return gr.update(), gr.update(), selected_indices
    
    preview_img = base_img.copy() # type: ignore
    
    if candidates:
        # Create two groups of masks
        selected_masks = []
        unselected_masks = []
        
        for i, cand in enumerate(candidates):
            if i in selected_indices:
                selected_masks.append(cand.binary_mask)
            else:
                unselected_masks.append(cand.binary_mask)
        
        if unselected_masks:
            preview_img = apply_mask_overlay(preview_img, np.array(unselected_masks), opacity=0.1)
        if selected_masks:
            preview_img = apply_mask_overlay(preview_img, np.array(selected_masks), opacity=0.6)
            
    # Update Gallery Captions
    updated_gallery = []
    for i, cand in enumerate(candidates):
        crop = create_mask_crop(base_img, cand.binary_mask)
        prefix = "✅ " if i in selected_indices else ""
        label = f"{prefix}{cand.class_name} ({cand.score:.2f})"
        updated_gallery.append((crop, label))
            
    return gr.update(value=preview_img), gr.update(value=updated_gallery), selected_indices

def select_all_candidates(candidates):
    """Select all candidates."""
    if not candidates: return gr.update(), gr.update(), set()
    
    all_indices = set(range(len(candidates)))
    
    # Trigger update logic (reuse on_gallery_select logic or duplicate?)
    # Let's just call the update logic manually
    base_img = controller.current_image
    if base_img is None: return gr.update(), gr.update(), all_indices
    
    preview_img = base_img.copy() # type: ignore
    
    all_masks = [c.binary_mask for c in candidates]
    if all_masks:
        preview_img = apply_mask_overlay(preview_img, np.array(all_masks), opacity=0.6)
        
    updated_gallery = []
    for i, cand in enumerate(candidates):
        crop = create_mask_crop(base_img, cand.binary_mask)
        label = f"✅ {cand.class_name} ({cand.score:.2f})"
        updated_gallery.append((crop, label))
        
    return gr.update(value=preview_img), gr.update(value=updated_gallery), all_indices

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

def format_candidate_list(candidates):
    choices = []
    for i, cand in enumerate(candidates):
        choices.append((f"Mask {i}: {cand.class_name} ({cand.score:.2f})", i))
    return choices

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

/* Gallery Adjustments */
.candidate-gallery { min-height: 300px; }
.candidate-gallery .grid-wrap { overflow-x: hidden !important; }
.candidate-gallery .gallery-item { padding: 2px !important; }

/* Hide Footer */
footer { display: none !important; }

/* Ensure the tab bar is a flex container and takes full width */
.main-tabs > div:first-of-type {
    display: flex !important;
    width: 100% !important;
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
                        gr.Examples(
                            examples=["example_img/DEPAL1_2025-11-28_12-31-29.710_81822238-07b7-4b4a-830b-5ab0e5272dbb.Color.png"],
                            inputs=single_image_input
                        )

                start_btn = gr.Button("Start Annotation", variant="primary", interactive=False)

            # --- SCREEN 1: INPUT ---
            with gr.TabItem("Input", id=1) as input_screen:
                gr.Markdown("### Generate initial objects")
                
                # Navigation (Full Width)
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    nav_status = gr.Textbox(label="Status", show_label=False, value="0/0", interactive=False, scale=2)
                    next_btn = gr.Button("Next")
                
                # Prompt Row (Moved)
                with gr.Row():
                    txt_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. cat, car", show_label=True, scale=2)
                    txt_class_name = gr.Textbox(label="Class Name Override", placeholder="Optional", show_label=True, scale=1)

                with gr.Row():
                    # Left Column: Image
                    with gr.Column(scale=3):
                        img_input = gr.Image(
                            label="Current Image (Click 2 Points for Box)", 
                            type="pil", 
                            height=600,
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
                        
                        run_btn = gr.Button("Run Inference", variant="primary", size="lg")

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
                            elem_classes="zoom-image",
                            height=600
                        )
                        
                    with gr.Column(scale=1):
                        # Gallery of crops
                        results_gallery = gr.Gallery(label="Found Candidates (Click to Select)", columns=3, height=300, object_fit="contain", allow_preview=False, elem_classes="candidate-gallery")
                        
                        # Selection List
                        with gr.Row():
                            select_all_btn = gr.Button("Select All", size="sm", variant="secondary")
                            
                        with gr.Row():
                            confirm_btn = gr.Button("Add Selected to Store", variant="primary")
                        
                        with gr.Row():
                            add_more_btn = gr.Button("Add Selected & Generate More", variant="secondary")

            # --- SCREEN 3: EDITOR ---
            with gr.TabItem("Editor", id=3) as editor_screen:
                with gr.Row():
                    gr.Markdown("### Refine individual objects")
                    editor_img_counter = gr.Markdown("Image 0/0", elem_id="editor-img-counter")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Main interactive image for refinement
                        refine_image = gr.Image(
                            label="Click to Refine",
                            type="pil",
                            interactive=False,
                            height=600,
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
    
    def start_session():
        if not controller.project.playlist:
             raise gr.Error("No images loaded.")
        
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
        inputs=[],
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
        outputs=[preview_image, results_gallery, st_selected_indices]
    )
    
    # 3c. Gallery Select
    results_gallery.select(
        fn=on_gallery_select,
        inputs=[st_selected_indices, st_candidates],
        outputs=[preview_image, results_gallery, st_selected_indices]
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
        outputs=[st_candidates, st_current_image, tabs]
    ).then(
        fn=render_results_step2,
        inputs=[st_candidates, st_current_image],
        outputs=[results_gallery, preview_image, st_selected_indices]
    ).then(
        fn=get_image_counter,
        outputs=[result_img_counter]
    ).then(
        fn=lambda: gr.update(value="Run Inference", interactive=True),
        inputs=[],
        outputs=[run_btn]
    )
    
    txt_prompt.submit(
        fn=start_inference,
        inputs=[st_clean_input_image, txt_prompt, st_boxes],
        outputs=[run_btn]
    ).then(
        fn=run_inference_fn,
        inputs=[st_clean_input_image, txt_prompt, st_boxes, st_labels, txt_class_name, st_crop_box],
        outputs=[st_candidates, st_current_image, tabs]
    ).then(
        fn=render_results_step2,
        inputs=[st_candidates, st_current_image],
        outputs=[results_gallery, preview_image, st_selected_indices]
    ).then(
        fn=get_image_counter,
        outputs=[result_img_counter]
    ).then(
        fn=lambda: gr.update(value="Run Inference", interactive=True),
        inputs=[],
        outputs=[run_btn]
    )

    txt_class_name.submit(
        fn=start_inference,
        inputs=[st_clean_input_image, txt_prompt, st_boxes],
        outputs=[run_btn]
    ).then(
        fn=run_inference_fn,
        inputs=[st_clean_input_image, txt_prompt, st_boxes, st_labels, txt_class_name, st_crop_box],
        outputs=[st_candidates, st_current_image, tabs]
    ).then(
        fn=render_results_step2,
        inputs=[st_candidates, st_current_image],
        outputs=[results_gallery, preview_image, st_selected_indices]
    ).then(
        fn=get_image_counter,
        outputs=[result_img_counter]
    ).then(
        fn=lambda: gr.update(value="Run Inference", interactive=True),
        inputs=[],
        outputs=[run_btn]
    )
    
    # Navigation to Export (Manual)
    # finish_export_btn handles this now
    
    # Load JS
    demo.load(None, None, None, js=custom_js)

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=app_theme, ssr_mode=False, mcp_server=True, show_error=True)
