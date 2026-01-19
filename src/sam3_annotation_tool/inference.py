import torch
import numpy as np
from PIL import Image
import warnings
import logging
from transformers import (
    Sam3Model, Sam3Processor, # type: ignore
    Sam3TrackerModel, Sam3TrackerProcessor, # type: ignore
    logging as transformers_logging
)
from .schemas import ObjectState, SelectorInput
from typing import Optional, Any

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The OrderedVocab you are attempting to save contains holes.*")
warnings.filterwarnings("ignore", message=".*You are using a model of type sam3_video to instantiate a model of type sam3_tracker.*")
transformers_logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global Models (loaded once)
_IMG_MODEL: Optional[Any] = None
_IMG_PROCESSOR: Optional[Any] = None
_TRK_MODEL: Optional[Any] = None
_TRK_PROCESSOR: Optional[Any] = None

def load_models():
    global _IMG_MODEL, _IMG_PROCESSOR, _TRK_MODEL, _TRK_PROCESSOR
    if _IMG_MODEL is not None: return

    print(f"🖥️  Using compute device: {device}")
    print("⏳ Loading SAM3 Models...")
    
    # Use local_files_only=True to skip network checks (faster, more consistent)
    local_only = True
    
    try:
        # 1. Selector (Sam3Model)
        _IMG_MODEL = Sam3Model.from_pretrained("facebook/sam3", local_files_only=local_only).to(device)
        _IMG_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=local_only)
        
        # 2. Refiner (Sam3TrackerModel)
        _TRK_MODEL = Sam3TrackerModel.from_pretrained("facebook/sam3", local_files_only=local_only).to(device)
        _TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained("facebook/sam3", local_files_only=local_only)
        
    except OSError:
        # Models not cached, need to download first
        print(f"⚠️  Models not in cache, downloading... (this only happens once)")
        _IMG_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(device)
        _IMG_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
        _TRK_MODEL = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
        _TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    
    print(f"✅ All models loaded!")

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
    assert _IMG_MODEL is not None
    assert _IMG_PROCESSOR is not None
    
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
            print(f"✂️  Cropped image to: {image.size} (Offset: {crop_offset_x}, {crop_offset_y})")
    
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
    print(f"   - Box Labels: {input_labels if 'input_labels' in locals() else 'None'}")
    print(f"   - Image Size: {image.size}")
        
    # Note: Sam3Processor might not support input_labels directly in the same way as input_boxes for prompt encoding
    # If the model supports it, we should pass it. If not, we might need to filter boxes manually or check documentation.
    # Assuming standard SAM-like behavior where boxes don't usually have labels in this specific API call unless it's point prompts.
    # However, for "Include/Exclude" areas, if the model treats all boxes as "Include", we have a problem.
    # Let's check if we can pass it.
    
    # SAM3 requires input_ids even if only using boxes. 
    # If no text is provided, we pass a dummy empty string to generate padding tokens.
    # text_input = [selector_input.text] if (selector_input.text and selector_input.text.strip()) else [""]
    
    # User instruction: Pass [None] if text is empty, do not pass "" or None
    # text_input = [selector_input.text] if (selector_input.text and selector_input.text.strip()) else [None]

    # Construct arguments dynamically to omit missing inputs
    processor_kwargs = {
        "images": image,
        "return_tensors": "pt"
    }
    
    if selector_input.text and selector_input.text.strip():
        processor_kwargs["text"] = [selector_input.text]
        
    if input_boxes is not None:
        processor_kwargs["input_boxes"] = input_boxes
        
    if input_labels is not None:
        processor_kwargs["input_boxes_labels"] = input_labels

    inputs = _IMG_PROCESSOR(**processor_kwargs).to(device)
    
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
        
        final_name = selector_input.class_name_override or selector_input.text or "Object"
        
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
    print(f"🔧 Refine Inputs:")
    print(f"   - Anchor Box: {obj_state.anchor_box}")
    print(f"   - Points: {obj_state.input_points}")
    print(f"   - Point Labels: {obj_state.input_labels}")

    if _TRK_MODEL is None: load_models()
    assert _TRK_MODEL is not None
    assert _TRK_PROCESSOR is not None
    
    original_w, original_h = image.size
    image = image.convert("RGB")
    
    # --- Dynamic Cropping Logic ---
    # 1. Determine bounding box of interest (Anchor Box + All Input Points)
    # This is the "Refinement Box" that encompasses the object and new points
    rx1, ry1, rx2, ry2 = obj_state.anchor_box
    
    if obj_state.input_points:
        for pt in obj_state.input_points:
            px, py = pt
            rx1 = min(rx1, px)
            ry1 = min(ry1, py)
            rx2 = max(rx2, px)
            ry2 = max(ry2, py)
            
    # 2. Add Padding (25%) to create the Crop Box
    width = rx2 - rx1
    height = ry2 - ry1
    padding = int(max(width, height) * 0.25)
    
    cx1 = max(0, int(rx1 - padding))
    cy1 = max(0, int(ry1 - padding))
    cx2 = min(original_w, int(rx2 + padding))
    cy2 = min(original_h, int(ry2 + padding))
    
    crop_offset_x, crop_offset_y = cx1, cy1
    
    # 3. Crop Image
    if cx2 > cx1 and cy2 > cy1:
        image = image.crop((cx1, cy1, cx2, cy2))
    else:
        crop_offset_x, crop_offset_y = 0, 0
        
    # --- Coordinate Adjustment ---
    
    # Use the Refinement Box (tight) as the prompt, adjusted to crop coordinates
    box_float = [
        float(rx1 - crop_offset_x), 
        float(ry1 - crop_offset_y), 
        float(rx2 - crop_offset_x), 
        float(ry2 - crop_offset_y)
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
