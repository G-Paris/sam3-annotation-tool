import torch
import spaces
import gradio as gr
import numpy as np
from PIL import Image
from transformers import (
    Sam3Model, Sam3Processor,
    Sam3TrackerModel, Sam3TrackerProcessor
)
from .utils import apply_mask_overlay, draw_points_on_image, create_mask_crop

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_bbox_from_mask(mask_img):
    if mask_img is None: return None
    mask_arr = np.array(mask_img)
    # Check if empty
    if mask_arr.max() == 0: return None
    
    if mask_arr.ndim == 3:
        # If RGBA/RGB, usually the drawing is colored or white.
        # Let's take max across channels to be safe
        mask_arr = mask_arr.max(axis=2)
        
    y_indices, x_indices = np.where(mask_arr > 0)
    if len(y_indices) == 0: return None
    
    x1, x2 = np.min(x_indices), np.max(x_indices)
    y1, y2 = np.min(y_indices), np.max(y_indices)
    return [float(x1), float(y1), float(x2), float(y2)]

class Sam3Manager:
    def __init__(self):
        self.img_model = None
        self.img_processor = None
        self.trk_model = None
        self.trk_processor = None
        self.device = device

    def load_models(self):
        print(f"🖥️ Using compute device: {self.device}")
        print("⏳ Loading SAM3 Models...")
        try:
            # 1. Load Image Segmentation Model (Text)
            print("   ... Loading Image Text Model")
            self.img_model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            self.img_processor = Sam3Processor.from_pretrained("facebook/sam3")

            # 2. Load Image Tracker Model (Click)
            print("   ... Loading Image Tracker Model")
            self.trk_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(self.device)
            self.trk_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
            
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR LOADING MODELS: {e}")
            raise e

# Global instance
manager = Sam3Manager()

# We load models immediately when this module is imported/initialized
# In a real app, we might want to do this lazily or explicitly, but for now we follow the original pattern
try:
    manager.load_models()
except:
    pass # Errors handled in UI or logs

@spaces.GPU
def run_image_segmentation(source_img_input, class_names_str, conf_thresh=0.5):
    if manager.img_model is None or manager.img_processor is None:
        raise gr.Error("Models failed to load.")
        
    pil_image = None
    input_boxes = None
    
    # Handle ImageEditor Input (Dict) or Plain Image
    if isinstance(source_img_input, dict):
        # Gradio ImageEditor returns {'background': img, 'layers': [img], 'composite': img}
        pil_image = source_img_input.get('background')
        
        # If background is None (e.g. user just drew on blank?), try composite
        if pil_image is None:
            pil_image = source_img_input.get('composite')
            
        layers = source_img_input.get('layers')
        mask_img = None
        if layers and len(layers) > 0:
            # Use the first layer as the mask (assuming single drawing layer)
            mask_img = layers[0]
        
        # Try to get bbox from mask
        bbox = get_bbox_from_mask(mask_img)
        if bbox:
            input_boxes = [[bbox]] # Batch 1, 1 box
            print(f"Using drawn bbox: {bbox}")
    else:
        pil_image = source_img_input

    if pil_image is None:
        raise gr.Error("Please provide an image.")
    
    # Parse Class Names
    if not class_names_str:
        # Fallback or error? Let's allow empty if bbox is present, but usually we need a prompt.
        # If bbox is present but no text, SAM3 might need a generic prompt or handles None.
        # Let's assume user must provide at least one class or prompt.
        text_prompts = ["object"] # Default if empty?
    else:
        text_prompts = [x.strip() for x in class_names_str.split(",") if x.strip()]
        
    try:
        pil_image = pil_image.convert("RGB")
        
        # Processor call
        model_inputs = manager.img_processor(
            images=pil_image, 
            text=text_prompts, 
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(manager.device)

        with torch.no_grad():
            inference_output = manager.img_model(**model_inputs)

        processed_results = manager.img_processor.post_process_instance_segmentation(
            inference_output,
            threshold=conf_thresh,
            mask_threshold=0.5,
            target_sizes=model_inputs.get("original_sizes").tolist()
        )[0]

        annotation_list = []
        mask_data_list = []
        dropdown_choices = []
        
        raw_masks = processed_results['masks'].cpu().numpy()
        raw_scores = processed_results['scores'].cpu().numpy()
        
        raw_labels = processed_results.get('labels', None)
        
        # Handle dimensions if necessary (SAM3 usually returns [N, H, W])
        if raw_masks.ndim == 4: raw_masks = raw_masks.squeeze(1)
        
        for idx, mask_array in enumerate(raw_masks):
            score = float(raw_scores[idx])
            
            # Determine label name
            if raw_labels is not None:
                label_idx = raw_labels[idx]
                if label_idx < len(text_prompts):
                    label_name = text_prompts[label_idx]
                else:
                    label_name = f"Class {label_idx}"
            else:
                # If no labels returned, use the first prompt or generic
                label_name = text_prompts[0] if len(text_prompts) == 1 else "Detected Object"

            label_str = f"{label_name} ({score:.2f})"
            unique_id = f"{idx}: {label_str}"
            
            # For AnnotatedImage: (mask, label)
            annotation_list.append((mask_array, label_str))
            
            # For State
            mask_data_list.append({
                "id": idx,
                "unique_id": unique_id,
                "mask": mask_array, 
                "label": label_name,
                "score": score
            })
            
            # For Dropdown
            dropdown_choices.append(unique_id)
            
        return (pil_image, annotation_list), mask_data_list, gr.update(choices=dropdown_choices, value=None, visible=True)

    except Exception as e:
        raise gr.Error(f"Error during image processing: {e}")

@spaces.GPU
def run_image_click_gpu(input_image, x, y, points_state, labels_state):
    if manager.trk_model is None or manager.trk_processor is None:
        raise gr.Error("Tracker Model failed to load.")
    
    if input_image is None: return input_image, [], []
    if points_state is None: points_state = []; labels_state = []
    
    # Append new point
    points_state.append([x, y])
    labels_state.append(1) # 1 indicates a positive click (foreground)

    try:
        # Prepare inputs format: [Batch, Point_Group, Point_Idx, Coord]
        input_points = [[points_state]] 
        input_labels = [[labels_state]]
        
        inputs = manager.trk_processor(images=input_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(manager.device)
        
        with torch.no_grad():
            # multimask_output=True usually helps with ambiguity, but let's default to best mask for simplicity here
            outputs = manager.trk_model(**inputs, multimask_output=False)
            
        # Post process
        masks = manager.trk_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"], binarize=True)[0]
        
        # Overlay mask
        # masks shape is [1, 1, H, W] for single object tracking
        final_img = apply_mask_overlay(input_image, masks[0])
        
        # Draw the visual points on top
        final_img = draw_points_on_image(final_img, points_state)
        
        return final_img, points_state, labels_state

    except Exception as e:
        print(f"Tracker Error: {e}")
        return input_image, points_state, labels_state
