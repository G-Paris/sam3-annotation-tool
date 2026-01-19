import gradio as gr
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_image(image, boxes, labels, pending_point=None, crop_box=None):
    """Helper to draw boxes and pending point on image."""
    if image is None: return None
    out_img = image.copy()
    draw = ImageDraw.Draw(out_img)
    
    w, h = image.size
    
    # Draw existing boxes
    for box, label in zip(boxes, labels):
        color = "#00FF00" if label == 1 else "#FF0000" # Green for Include, Red for Exclude
        draw.rectangle(box, outline=color, width=3)
        
    # Draw crop box if exists
    if crop_box:
        draw.rectangle(crop_box, outline="blue", width=3)
        # Add label
        draw.text((crop_box[0], crop_box[1]-15), "CROP", fill="blue")
        
    # Draw pending point if exists
    if pending_point:
        x, y = pending_point
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), fill="yellow", outline="black")
        
        # Draw crosshair guides
        draw.line([(0, y), (w, y)], fill="cyan", width=1)
        draw.line([(x, 0), (x, h)], fill="cyan", width=1)
        
    return out_img

def format_box_list(boxes, labels):
    """Format boxes for display in Dataframe (Editable)."""
    data = []
    for i, box in enumerate(boxes):
        lbl = "Include" if labels[i] == 1 else "Exclude"
        # [Delete?, Type, x1, y1, x2, y2]
        data.append([False, lbl, box[0], box[1], box[2], box[3]])
    return data

def format_crop_box(crop_box):
    """Format crop box for display in Dataframe."""
    if not crop_box:
        return []
    # [Delete?, x1, y1, x2, y2]
    return [[False, crop_box[0], crop_box[1], crop_box[2], crop_box[3]]]
def draw_candidates(image: Image.Image, candidates: list, selected_indices: set | int | None = None):
    """
    Draws all candidates on the image with ID labels.
    - selected_indices: If provided (set, list, or int), highlights these candidates and dims others.
      If None, all are shown as active candidates.
    """
    if image is None: return None
    
    # Normalize selected_indices to a set or None
    if selected_indices is not None:
        if isinstance(selected_indices, int):
            selected_indices = {selected_indices}
        elif isinstance(selected_indices, list):
            selected_indices = set(selected_indices)
        elif not isinstance(selected_indices, set):
            # Fallback
            selected_indices = None
            
    # Work on RGBA for transparency
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

    for idx, obj in enumerate(candidates):
        if obj.binary_mask is None: continue
        
        # Determine style based on selection
        is_selected = (selected_indices is not None) and (idx in selected_indices)
        # If nothing is selected (None), all are "active". 
        # If something is selected, only selected ones are active/highlighted.
        is_active = (selected_indices is None) or is_selected
        
        if is_active:
            # Generate unique color for this index using Golden Ratio for distinctness
            hue = (idx * 0.618033988749895) % 1
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            base_rgb = (int(r*255), int(g*255), int(b*255))
            
            if selected_indices is None:
                 # Default candidate view - use unique colors
                 fill_color = (*base_rgb, 100) 
            else:
                 # Selected view - use unique colors (more opaque)
                 fill_color = (*base_rgb, 160) 
                 
            text_color = (255, 255, 255, 255)
        else:
            # Dimmed Color (Grayed out)
            fill_color = (128, 128, 128, 30)
            text_color = (200, 200, 200, 100)

        # 1. Draw Mask
        # Create a mask image for this object
        mask_uint8 = (obj.binary_mask * 255).astype(np.uint8)
        mask_layer = Image.fromarray(mask_uint8, mode='L')
        
        # Colorize mask
        colored_mask = Image.new("RGBA", canvas.size, fill_color)
        overlay.paste(colored_mask, (0, 0), mask_layer)
        
        # 2. Draw ID at Centroid
        y_indices, x_indices = np.where(obj.binary_mask)
        if len(y_indices) > 0:
            cy = int(np.mean(y_indices))
            cx = int(np.mean(x_indices))
            
            label = str(idx + 1)
            
            # Draw text background for readability
            bbox = draw.textbbox((cx, cy), label, font=font, anchor="mm")
            # Add padding
            draw.rectangle([bbox[0]-4, bbox[1]-4, bbox[2]+4, bbox[3]+4], fill=(0, 0, 0, 160))
            draw.text((cx, cy), label, font=font, fill=text_color, anchor="mm")

    # Composite
    return Image.alpha_composite(canvas, overlay).convert("RGB")
def parse_dataframe(df_data):
    """Parse dataframe back to boxes and labels."""
    boxes = []
    labels = []
    
    # Handle if df_data is None or empty
    if df_data is None:
        return [], []
        
    # Check if it's a pandas DataFrame
    if hasattr(df_data, 'values'):
        if df_data.empty:
            return [], []
        values = df_data.values.tolist()
    else:
        if not df_data:
            return [], []
        values = df_data

    for row in values:
        # row[0] is Delete? (bool)
        # row[1] is Type (str)
        # row[2-5] are coords
        
        lbl = 1 if row[1] == "Include" else 0
        try:
            # Ensure coords are ints
            box = [int(float(row[2])), int(float(row[3])), int(float(row[4])), int(float(row[5]))]
            boxes.append(box)
            labels.append(lbl)
        except (ValueError, TypeError, IndexError):
            continue # Skip invalid rows
            
    return boxes, labels

def parse_crop_dataframe(df_data):
    """Parse dataframe back to crop box."""
    if df_data is None: return None
    
    values = []
    if hasattr(df_data, 'values'):
        if df_data.empty: return None
        values = df_data.values.tolist()
    else:
        values = df_data
        
    if not values: return None
    
    # Take the first valid row
    for row in values:
        # row[0] is Delete?
        if row[0]: return None # Deleted
        
        try:
            # row[1-4] are coords (since no Type column)
            box = [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4]))]
            return box
        except:
            continue
            
    return None

def on_dataframe_change(df_data, clean_img, crop_box):
    """Handle changes in the dataframe (edits)."""
    if clean_img is None: return gr.update(), [], []
    
    boxes, labels = parse_dataframe(df_data)
    vis_img = draw_boxes_on_image(clean_img, boxes, labels, None, crop_box)
    
    return vis_img, boxes, labels

def on_crop_dataframe_change(df_data, clean_img, boxes, labels):
    """Handle changes in the crop dataframe."""
    if clean_img is None: return gr.update(), None
    
    crop_box = parse_crop_dataframe(df_data)
    vis_img = draw_boxes_on_image(clean_img, boxes, labels, None, crop_box)
    
    return vis_img, crop_box

def delete_checked_boxes(df_data, clean_img, crop_box):
    """Delete boxes that are checked."""
    if clean_img is None: return [], [], gr.update(), gr.update()
    
    new_boxes = []
    new_labels = []
    
    values = []
    if df_data is not None:
        if hasattr(df_data, 'values'):
             values = df_data.values.tolist()
        else:
             values = df_data
    
    # Filter
    if values:
        for row in values:
            is_deleted = row[0]
            if not is_deleted:
                lbl = 1 if row[1] == "Include" else 0
                try:
                    box = [int(float(row[2])), int(float(row[3])), int(float(row[4])), int(float(row[5]))]
                    new_boxes.append(box)
                    new_labels.append(lbl)
                except:
                    pass

    vis_img = draw_boxes_on_image(clean_img, new_boxes, new_labels, None, crop_box)
    new_df = format_box_list(new_boxes, new_labels)
    
    return new_boxes, new_labels, new_df, vis_img

def on_upload(files):
    """Handle image upload (list of files)."""
    if not files:
        return None, [], [], None
        
    # files is a list of file paths (strings) or file objects depending on Gradio version/config
    # With file_count="multiple", it's usually a list of temp paths.
    
    # If it's a single file (legacy check), wrap it
    if not isinstance(files, list):
        files = [files]
        
    # Extract paths
    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, 'name'):
            paths.append(f.name)
            
    # Import controller inside function to avoid circular import
    from .controller import controller
    first_image = controller.load_playlist(paths)
    
    return first_image, [], [], None # clean_img, boxes, labels, pending_pt

def on_input_image_select(evt: gr.SelectData, pending_pt, boxes, labels, click_effect, clean_img, crop_box):
    """Handle click on input image to define boxes or crop."""
    if clean_img is None: return gr.update(), pending_pt, boxes, labels, gr.update(), crop_box, gr.update()
    
    x, y = evt.index
    
    if pending_pt is None:
        # First point
        new_pending = (x, y)
        # Draw point
        vis_img = draw_boxes_on_image(clean_img, boxes, labels, new_pending, crop_box)
        return vis_img, new_pending, boxes, labels, gr.update(), crop_box, gr.update()
    else:
        # Second point - Finalize box or crop
        x1, y1 = pending_pt
        x2, y2 = x, y
        
        # Create box [x_min, y_min, x_max, y_max]
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        
        if click_effect == "Crop Initial Image":
            # Update crop box (overwrite)
            new_crop_box = bbox
            vis_img = draw_boxes_on_image(clean_img, boxes, labels, None, new_crop_box)
            new_crop_df = format_crop_box(new_crop_box)
            return vis_img, None, boxes, labels, gr.update(), new_crop_box, new_crop_df
        else:
            # Add to list (Include/Exclude)
            lbl = 1 if click_effect == "Include Area" else 0
            new_boxes = boxes + [bbox]
            new_labels = labels + [lbl]
            
            # Draw all
            vis_img = draw_boxes_on_image(clean_img, new_boxes, new_labels, None, crop_box)
            
            # Update dataframe
            new_df = format_box_list(new_boxes, new_labels)
            
            return vis_img, None, new_boxes, new_labels, new_df, crop_box, gr.update()

def undo_last_click(pending_pt, boxes, labels, clean_img, crop_box):
    """Undo the last click or remove the last box."""
    if clean_img is None: return gr.update(), None, boxes, labels, gr.update(), crop_box, gr.update()
    
    # Case 1: Pending point exists (user clicked once) -> Clear it
    if pending_pt is not None:
        # Redraw only boxes
        vis_img = draw_boxes_on_image(clean_img, boxes, labels, None, crop_box)
        return vis_img, None, boxes, labels, gr.update(), crop_box, gr.update()
    
    # Case 2: No pending point, but boxes exist -> Remove last box
    # Note: We don't undo crop box here easily unless we track history. 
    # For now, let's assume undo only affects boxes stack.
    if boxes:
        boxes.pop()
        labels.pop()
        vis_img = draw_boxes_on_image(clean_img, boxes, labels, None, crop_box)
        new_df = format_box_list(boxes, labels)
        return vis_img, None, boxes, labels, new_df, crop_box, gr.update()
        
    # Case 3: Nothing to undo
    return gr.update(), None, boxes, labels, gr.update(), crop_box, gr.update()
