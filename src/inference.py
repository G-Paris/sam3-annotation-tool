import torch
import numpy as np
from PIL import Image
import warnings
import logging
from transformers import (
    Sam3Model, Sam3Processor,
    Sam3TrackerModel, Sam3TrackerProcessor,
    logging as transformers_logging
)
from .schemas import ObjectState, SelectorInput

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The OrderedVocab you are attempting to save contains holes.*")
warnings.filterwarnings("ignore", message=".*You are using a model of type sam3_video to instantiate a model of type sam3_tracker.*")
transformers_logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global Models (loaded once)
_IMG_MODEL = None
_IMG_PROCESSOR = None
_TRK_MODEL = None
_TRK_PROCESSOR = None

def load_models():
    global _IMG_MODEL, _IMG_PROCESSOR, _TRK_MODEL, _TRK_PROCESSOR
    if _IMG_MODEL is not None: return

    print(f"🖥️ Using compute device: {device}")
    print("⏳ Loading SAM3 Models...")
    
    # 1. Selector (Sam3Model)
    _IMG_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(device)
    _IMG_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
    
    # 2. Refiner (Sam3TrackerModel)
    _TRK_MODEL = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
    _TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    
    print("✅ Models loaded successfully!")

def get_bbox_from_mask(mask_arr):
    if mask_arr is None: return None
    if mask_arr.max() == 0: return None
    
    y_indices, x_indices = np.where(mask_arr > 0)
    if len(y_indices) == 0: return None
    
    x1, x2 = np.min(x_indices), np.max(x_indices)
    y1, y2 = np.min(y_indices), np.max(y_indices)
    # Cast to int for schema compatibility
    return [int(x1), int(y1), int(x2), int(y2)]

def search_objects(selector_input: SelectorInput) -> list[ObjectState]:
    """
    Stage A: The Selector
    """
    if _IMG_MODEL is None: load_models()
    
    image = selector_input.image.convert("RGB")
    original_w, original_h = image.size
    
    # Handle Cropping
    crop_offset_x, crop_offset_y = 0, 0
    if selector_input.crop_box:
        cx1, cy1, cx2, cy2 = selector_input.crop_box
        
        # Ensure valid crop within image bounds
        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(original_w, cx2)
        cy2 = min(original_h, cy2)
        
        if cx2 > cx1 and cy2 > cy1:
            image = image.crop((cx1, cy1, cx2, cy2))
            crop_offset_x, crop_offset_y = cx1, cy1
            print(f"✂️ Cropped image to: {image.size} (Offset: {crop_offset_x}, {crop_offset_y})")
    
    # Prepare inputs
    input_boxes = None
    input_labels = None
    
    if selector_input.input_boxes:
        # Adjust boxes to crop coordinates
        adjusted_boxes = []
        for box in selector_input.input_boxes:
            bx1, by1, bx2, by2 = box
            # Subtract offset
            bx1 -= crop_offset_x
            by1 -= crop_offset_y
            bx2 -= crop_offset_x
            by2 -= crop_offset_y
            # Clip to crop bounds (0 to crop_w/h)
            crop_w, crop_h = image.size
            bx1 = max(0, min(crop_w, bx1))
            by1 = max(0, min(crop_h, by1))
            bx2 = max(0, min(crop_w, bx2))
            by2 = max(0, min(crop_h, by2))
            
            adjusted_boxes.append([float(bx1), float(by1), float(bx2), float(by2)])
            
        # SAM3 expects [[ [x1, y1, x2, y2], ... ]] for batch size 1
        input_boxes = [adjusted_boxes]
        
        if selector_input.input_labels:
             # Shape: (Batch, N_boxes) -> [[1, 0, ...]]
             input_labels = [selector_input.input_labels]
    
    print(f"🔍 Search Inputs:")
    print(f"   - Text: '{selector_input.text}'")
    print(f"   - Boxes: {input_boxes}")
    print(f"   - Image Size: {image.size}")
        
    # Note: Sam3Processor might not support input_labels directly in the same way as input_boxes for prompt encoding
    # If the model supports it, we should pass it. If not, we might need to filter boxes manually or check documentation.
    # Assuming standard SAM-like behavior where boxes don't usually have labels in this specific API call unless it's point prompts.
    # However, for "Include/Exclude" areas, if the model treats all boxes as "Include", we have a problem.
    # Let's check if we can pass it.
    
    inputs = _IMG_PROCESSOR(
        images=image, 
        text=[selector_input.text], 
        input_boxes=input_boxes,
        input_boxes_labels=input_labels, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = _IMG_MODEL(**inputs)
        
    results = _IMG_PROCESSOR.post_process_instance_segmentation(
        outputs, 
        threshold=0.4, # Configurable?
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    candidates = []
    raw_masks = results['masks'].cpu().numpy() # [N, H, W] or [N, 1, H, W]
    raw_scores = results['scores'].cpu().numpy()
    
    if raw_masks.ndim == 4: raw_masks = raw_masks.squeeze(1)
    
    for idx, mask in enumerate(raw_masks):
        # mask is boolean/binary for the CROPPED image
        
        # Restore to full size if cropped
        if selector_input.crop_box:
            full_mask = np.zeros((original_h, original_w), dtype=bool)
            # Paste cropped mask back
            # mask shape is (crop_h, crop_w)
            mh, mw = mask.shape
            full_mask[crop_offset_y:crop_offset_y+mh, crop_offset_x:crop_offset_x+mw] = mask
            mask = full_mask
            
        anchor_box = get_bbox_from_mask(mask)
        if anchor_box is None: continue
        
        final_name = selector_input.class_name_override if selector_input.class_name_override else selector_input.text
        
        candidates.append(ObjectState(
            score=float(raw_scores[idx]),
            anchor_box=anchor_box,
            binary_mask=mask,
            initial_mask=mask, # Save copy for undo
            class_name=final_name
        ))
        
    return candidates

def refine_object(image: Image.Image, obj_state: ObjectState) -> np.ndarray:
    """
    Stage B: The Refiner
    """
    if _TRK_MODEL is None: load_models()
    
    original_w, original_h = image.size
    image = image.convert("RGB")
    
    # --- Dynamic Cropping Logic ---
    # 1. Determine bounding box of interest (Anchor Box + All Input Points)
    x1, y1, x2, y2 = obj_state.anchor_box
    
    if obj_state.input_points:
        for pt in obj_state.input_points:
            px, py = pt
            x1 = min(x1, px)
            y1 = min(y1, py)
            x2 = max(x2, px)
            y2 = max(y2, py)
            
    # 2. Add Padding (25%)
    width = x2 - x1
    height = y2 - y1
    padding = int(max(width, height) * 0.25)
    
    cx1 = max(0, int(x1 - padding))
    cy1 = max(0, int(y1 - padding))
    cx2 = min(original_w, int(x2 + padding))
    cy2 = min(original_h, int(y2 + padding))
    
    crop_offset_x, crop_offset_y = cx1, cy1
    
    # 3. Crop Image
    if cx2 > cx1 and cy2 > cy1:
        image = image.crop((cx1, cy1, cx2, cy2))
        # print(f"✂️ Refiner Cropped to: {image.size} (Offset: {crop_offset_x}, {crop_offset_y})")
    else:
        # Fallback if invalid crop (shouldn't happen)
        crop_offset_x, crop_offset_y = 0, 0
        
    # --- Coordinate Adjustment ---
    
    # Adjust Anchor Box
    ax1, ay1, ax2, ay2 = obj_state.anchor_box
    box_float = [
        float(ax1 - crop_offset_x), 
        float(ay1 - crop_offset_y), 
        float(ax2 - crop_offset_x), 
        float(ay2 - crop_offset_y)
    ]
    
    # Adjust Points
    points_float = []
    for p in obj_state.input_points:
        points_float.append([float(p[0] - crop_offset_x), float(p[1] - crop_offset_y)])
    
    # Prepare inputs
    input_boxes = [[box_float]]
    
    # Nesting for Sam3TrackerProcessor:
    # input_points: 4 levels [Image, Object, Point, Coords]
    # input_labels: 3 levels [Image, Object, Label]
    
    # obj_state.input_points is List[List[float]] (Points for 1 object) -> Level 3 & 4
    # So we need to wrap it in [ [ ... ] ] for Image and Object levels
    input_points = [[points_float]] 
    
    # obj_state.input_labels is List[int] (Labels for 1 object) -> Level 3
    # So we need to wrap it in [ [ ... ] ] for Image and Object levels
    input_labels = [[obj_state.input_labels]]
    
    inputs = _TRK_PROCESSOR(
        images=image,
        input_boxes=input_boxes,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = _TRK_MODEL(**inputs, multimask_output=False)
        
    masks = _TRK_PROCESSOR.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"], 
        binarize=True
    )[0]
    
    final_mask_crop = masks[0].numpy()
    if final_mask_crop.ndim == 3: final_mask_crop = final_mask_crop[0]
    
    # --- Restore Mask to Full Size ---
    final_mask = np.zeros((original_h, original_w), dtype=bool)
    
    mh, mw = final_mask_crop.shape
    final_mask[crop_offset_y:crop_offset_y+mh, crop_offset_x:crop_offset_x+mw] = final_mask_crop
    
    return final_mask
