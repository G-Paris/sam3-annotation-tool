import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from src.theme import CustomBlueTheme
from src.controller import controller
from src.inference import load_models
from src.utils import apply_mask_overlay, draw_points_on_image, get_bbox_from_mask, create_mask_crop

# Load models immediately on startup
load_models()

app_theme = CustomBlueTheme()

# --- Helper Functions ---

def process_upload(image_input):
    """Handle image upload."""
    if not image_input: return None
    image = image_input.get("background")
    if image is None: image = image_input.get("composite")
    if image: controller.set_image(image)
    return image

def add_box_from_drawing(image_input, box_type, current_boxes, current_labels):
    """Extract box from drawing and add to list."""
    if not image_input: return current_boxes, current_labels, image_input
    
    layers = image_input.get("layers")
    if not layers or len(layers) == 0:
        return current_boxes, current_labels, image_input # No drawing
        
    # Get bbox from first layer
    bbox = get_bbox_from_mask(layers[0])
    if not bbox:
        return current_boxes, current_labels, image_input
        
    # Add to lists
    label = 1 if box_type == "Include Area" else 0
    
    new_boxes = current_boxes + [bbox]
    new_labels = current_labels + [label]
    
    # Clear drawing by returning original background as new value
    # ImageEditor expects {background: img, layers: [], composite: img}
    # But passing just the background image usually resets it
    background = image_input.get("background")
    
    return new_boxes, new_labels, background

def format_box_list(boxes, labels):
    """Format boxes for display in Dataframe."""
    data = []
    for i, box in enumerate(boxes):
        lbl = "Include" if labels[i] == 1 else "Exclude"
        data.append([i, lbl, str(box)])
    return data

def on_box_select(evt: gr.SelectData):
    """Enable delete button when row selected."""
    return evt.index[0], gr.update(interactive=True)

def delete_box_wrapper(idx, boxes, labels):
    """Delete box by index."""
    if idx is not None and 0 <= idx < len(boxes):
        boxes.pop(idx)
        labels.pop(idx)
    return boxes, labels, None, gr.update(interactive=False)

def run_inference_step1(image_input, text_prompt, boxes, labels):
    """Step 1: Run Inference and switch screens."""
    print(f"🖱️ Run Inference Clicked! Prompt: '{text_prompt}', Boxes: {len(boxes)}")
    
    if not image_input: 
        raise gr.Error("Please upload an image.")
    if not text_prompt: 
        raise gr.Error("Please enter a text prompt.")
    
    image = image_input.get("background") or image_input.get("composite")
    if image is None:
        raise gr.Error("Failed to process image.")
        
    controller.set_image(image)
    
    try:
        candidates = controller.search_and_add(text_prompt, boxes, labels)
        print(f"✅ Search returned {len(candidates)} candidates.")
    except Exception as e:
        print(f"❌ Error during search: {e}")
        raise gr.Error(f"Inference failed: {str(e)}")
        
    # Return candidates, image, and screen visibility updates
    return (
        candidates,
        image,
        gr.update(visible=False), # Hide Input
        gr.update(visible=True)   # Show Results
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
    
    preview_img = base_img.copy()
    
    # We want to show ALL masks, but highlight selected ones
    # Or just show selected ones? User said "highlight the mask in the overview image when selecting"
    # Let's show selected ones with high opacity, others with very low opacity
    
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
    preview_img = base_img.copy()
    
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

def export_results():
    """Export results to output folder."""
    try:
        res = controller.export_data("output")
        if res:
            _, txt_path = res
            return f"Exported annotations to {txt_path}"
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
    
    return "Added to Store!", gr.update(visible=False), gr.update(visible=True) # Go to Editor?

# --- UI Layout ---

custom_css="""
#col-container { margin: 0 auto; max-width: 1100px; }
#main-title h1 { font-size: 2.1em !important; }
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
    
    # Hidden status box for messages
    status_box = gr.Textbox(visible=False)
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **SAM3 Annotator**", elem_id="main-title")
        
        # --- SCREEN 1: INPUT ---
        with gr.Column(visible=True) as input_screen:
            with gr.Row():
                with gr.Column(scale=2):
                    img_editor = gr.ImageEditor(
                        label="1. Upload Image & Draw Boxes", 
                        type="pil", 
                        height=500,
                        brush=gr.Brush(colors=["#FF0000"], color_mode="fixed")
                    )
                    
                    with gr.Row():
                        box_type = gr.Radio(["Include Area", "Exclude Area"], value="Include Area", label="Box Type")
                        add_box_btn = gr.Button("Add Box from Drawing", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Prompt Settings")
                    txt_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. cat, car")
                    
                    gr.Markdown("### 3. Box List")
                    box_list_display = gr.Dataframe(
                        headers=["Index", "Type", "Coordinates"], 
                        datatype=["number", "str", "str"],
                        interactive=False,
                        label="Added Boxes"
                    )
                    delete_box_btn = gr.Button("Delete Selected Box", variant="stop", interactive=False)
                    
                    run_btn = gr.Button("Run Inference", variant="primary", size="lg")

        # --- SCREEN 2: RESULTS ---
        with gr.Column(visible=False) as result_screen:
            gr.Markdown("### Inference Results")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Preview Image with ALL masks
                    preview_image = gr.Image(label="Selected Candidates Preview", type="pil", interactive=False)
                    
                with gr.Column(scale=1):
                    # Gallery of crops
                    results_gallery = gr.Gallery(label="Found Candidates (Click to Select)", columns=3, height=300, object_fit="contain", allow_preview=False)
                    
                    # Selection List
                    with gr.Row():
                        select_all_btn = gr.Button("Select All", size="sm", variant="secondary")
                        
                    with gr.Row():
                        confirm_btn = gr.Button("Add Selected to Store", variant="primary")

        # --- SCREEN 3: EDITOR ---
        with gr.Column(visible=False) as editor_screen:
            gr.Markdown("### 3. Refine Masks")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Main interactive image for refinement
                    refine_image = gr.Image(
                        label="Click to Refine",
                        type="pil",
                        interactive=True,
                        height=600
                    )
                    
                    with gr.Row():
                        undo_btn = gr.Button("Undo Last Click", variant="secondary")
                        # Explicit Radio for Mode
                        click_mode = gr.Radio(["Include (Green)", "Exclude (Red)"], value="Include (Green)", label="Click Mode", interactive=True)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Objects in Store")
                    
                    with gr.Row():
                        object_list = gr.Radio(
                            label="Select Object",
                            choices=[],
                            interactive=True
                        )
                    
                    with gr.Row():
                        revert_btn = gr.Button("Revert", size="sm", variant="secondary") # Icon replacement
                        delete_btn = gr.Button("Delete", size="sm", variant="stop")      # Icon replacement
                    
                    gr.Markdown("### Actions")
                    export_btn = gr.Button("Export Results (YOLO)", variant="primary")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
                    
                    # restart_btn = gr.Button("Start Over", variant="secondary")

    # --- Helper Functions for Editor ---
    
    def init_editor(selected_obj_id=None):
        """Initialize editor screen with current image and objects."""
        base_img = controller.current_image
        if base_img is None: return None, gr.update(choices=[])
        
        # Create choices for Radio
        choices = []
        for obj_id, obj in controller.store.objects.items():
            choices.append((f"{obj.class_name} ({obj_id[:4]}...)", obj_id))
            
        # Determine selection
        if selected_obj_id is None and choices:
            selected_obj_id = choices[0][1]
        elif selected_obj_id and selected_obj_id not in [c[1] for c in choices]:
             selected_obj_id = choices[0][1] if choices else None

        # Create overlay
        overlay_img = base_img.copy()
        
        if selected_obj_id and selected_obj_id in controller.store.objects:
            # Show ONLY selected object
            obj = controller.store.objects[selected_obj_id]
            mask = obj.binary_mask
            overlay_img = apply_mask_overlay(base_img, np.array([mask]), opacity=0.6)
            
            # Draw Points
            from PIL import ImageDraw
            draw = ImageDraw.Draw(overlay_img)
            radius = 5
            for pt, lbl in zip(obj.input_points, obj.input_labels):
                color = "#00FF00" if lbl == 1 else "#FF0000"
                x, y = pt
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white")
        
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
    
    # 1. Add Box
    add_box_btn.click(
        fn=add_box_from_drawing,
        inputs=[img_editor, box_type, st_boxes, st_labels],
        outputs=[st_boxes, st_labels, img_editor]
    ).then(
        fn=format_box_list,
        inputs=[st_boxes, st_labels],
        outputs=[box_list_display]
    )
    
    # 2. Delete Box
    box_list_display.select(
        fn=on_box_select,
        inputs=[],
        outputs=[st_selected_box_index, delete_box_btn]
    )
    
    delete_box_btn.click(
        fn=delete_box_wrapper,
        inputs=[st_selected_box_index, st_boxes, st_labels],
        outputs=[st_boxes, st_labels, st_selected_box_index, delete_box_btn]
    ).then(
        fn=format_box_list,
        inputs=[st_boxes, st_labels],
        outputs=[box_list_display]
    )
    
    # 3. Run Inference
    run_btn.click(
        fn=lambda: gr.update(value="Running Inference...", interactive=False),
        inputs=[],
        outputs=[run_btn]
    ).then(
        fn=run_inference_step1,
        inputs=[img_editor, txt_prompt, st_boxes, st_labels],
        outputs=[st_candidates, st_current_image, input_screen, result_screen]
    ).then(
        fn=render_results_step2,
        inputs=[st_candidates, st_current_image],
        outputs=[results_gallery, preview_image, st_selected_indices]
    ).then(
        fn=lambda: gr.update(value="Run Inference", interactive=True),
        inputs=[],
        outputs=[run_btn]
    )
    
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
    
    # 4. Back Button (Removed)
    # back_btn.click(...)
    
    # 5. Confirm Selection -> Go to Editor
    confirm_btn.click(
        fn=add_to_store_wrapper,
        inputs=[st_candidates, st_selected_indices],
        outputs=[status_box, result_screen, editor_screen]
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
        inputs=[],
        outputs=[export_status]
    )
    
    # 7. Restart (Removed)
    # restart_btn.click(...)

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=app_theme, ssr_mode=False, mcp_server=True, show_error=True)
