import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from src.controller import controller

def draw_boxes_on_image(image, boxes, labels, pending_point=None):
    """Helper to draw boxes and pending point on image."""
    if image is None: return None
    out_img = image.copy()
    draw = ImageDraw.Draw(out_img)
    
    w, h = image.size
    
    # Draw existing boxes
    for box, label in zip(boxes, labels):
        color = "#00FF00" if label == 1 else "#FF0000" # Green for Include, Red for Exclude
        draw.rectangle(box, outline=color, width=3)
        
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

def on_dataframe_change(df_data, clean_img):
    """Handle changes in the dataframe (edits)."""
    if clean_img is None: return gr.update(), [], []
    
    boxes, labels = parse_dataframe(df_data)
    vis_img = draw_boxes_on_image(clean_img, boxes, labels, None)
    
    return vis_img, boxes, labels

def delete_checked_boxes(df_data, clean_img):
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

    vis_img = draw_boxes_on_image(clean_img, new_boxes, new_labels, None)
    new_df = format_box_list(new_boxes, new_labels)
    
    return new_boxes, new_labels, new_df, vis_img

def on_upload(image):
    """Handle image upload."""
    if image:
        controller.set_image(image)
    return image, [], [], None # clean_img, boxes, labels, pending_pt

def on_input_image_select(evt: gr.SelectData, pending_pt, boxes, labels, box_type, clean_img):
    """Handle click on input image to define boxes."""
    if clean_img is None: return gr.update(), pending_pt, boxes, labels, gr.update()
    
    x, y = evt.index
    
    if pending_pt is None:
        # First point
        new_pending = (x, y)
        # Draw point
        vis_img = draw_boxes_on_image(clean_img, boxes, labels, new_pending)
        return vis_img, new_pending, boxes, labels, gr.update()
    else:
        # Second point - Finalize box
        x1, y1 = pending_pt
        x2, y2 = x, y
        
        # Create box [x_min, y_min, x_max, y_max]
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        
        # Add to list
        lbl = 1 if box_type == "Include Area" else 0
        new_boxes = boxes + [bbox]
        new_labels = labels + [lbl]
        
        # Draw all
        vis_img = draw_boxes_on_image(clean_img, new_boxes, new_labels, None)
        
        # Update dataframe
        new_df = format_box_list(new_boxes, new_labels)
        
        return vis_img, None, new_boxes, new_labels, new_df

def undo_last_click(pending_pt, boxes, labels, clean_img):
    """Undo the last click or remove the last box."""
    if clean_img is None: return gr.update(), None, boxes, labels, gr.update()
    
    # Case 1: Pending point exists (user clicked once) -> Clear it
    if pending_pt is not None:
        # Redraw only boxes
        vis_img = draw_boxes_on_image(clean_img, boxes, labels, None)
        return vis_img, None, boxes, labels, gr.update()
    
    # Case 2: No pending point, but boxes exist -> Remove last box
    if boxes:
        boxes.pop()
        labels.pop()
        vis_img = draw_boxes_on_image(clean_img, boxes, labels, None)
        new_df = format_box_list(boxes, labels)
        return vis_img, None, boxes, labels, new_df
        
    # Case 3: Nothing to undo
    return gr.update(), None, boxes, labels, gr.update()
