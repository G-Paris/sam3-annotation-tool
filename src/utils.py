import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw

def apply_mask_overlay(base_image, mask_data, opacity=0.5):
    """Draws segmentation masks on top of an image."""
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")
    
    if mask_data is None or len(mask_data) == 0:
        return base_image.convert("RGB")
        
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.cpu().numpy()
    mask_data = mask_data.astype(np.uint8)
    
    # Handle dimensions
    if mask_data.ndim == 4: mask_data = mask_data[0] 
    if mask_data.ndim == 3 and mask_data.shape[0] == 1: mask_data = mask_data[0]
    
    num_masks = mask_data.shape[0] if mask_data.ndim == 3 else 1
    if mask_data.ndim == 2:
        mask_data = [mask_data]
        num_masks = 1

    try:
        color_map = matplotlib.colormaps["rainbow"].resampled(max(num_masks, 1))
    except AttributeError:
        import matplotlib.cm as cm
        color_map = cm.get_cmap("rainbow").resampled(max(num_masks, 1))
        
    rgb_colors = [tuple(int(c * 255) for c in color_map(i)[:3]) for i in range(num_masks)]
    composite_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    
    for i, single_mask in enumerate(mask_data):
        mask_bitmap = Image.fromarray((single_mask * 255).astype(np.uint8))
        if mask_bitmap.size != base_image.size:
            mask_bitmap = mask_bitmap.resize(base_image.size, resample=Image.NEAREST)
        
        fill_color = rgb_colors[i]
        color_fill = Image.new("RGBA", base_image.size, fill_color + (0,))
        mask_alpha = mask_bitmap.point(lambda v: int(v * opacity) if v > 0 else 0)
        color_fill.putalpha(mask_alpha)
        composite_layer = Image.alpha_composite(composite_layer, color_fill)
        
    return Image.alpha_composite(base_image, composite_layer).convert("RGB")

def create_mask_crop(base_image, mask_array):
    """Creates a cropped image of the masked area on a transparent background."""
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")
    
    # Ensure mask is uint8
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8, mode='L')
    
    if mask_img.size != base_image.size:
        mask_img = mask_img.resize(base_image.size, resample=Image.NEAREST)
        
    # Create composite
    # We want the original image where the mask is, and transparent elsewhere
    result = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    result.paste(base_image, (0, 0), mask_img)
    
    # Crop to bounding box
    bbox = mask_img.getbbox()
    if bbox:
        result = result.crop(bbox)
        
    return result

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
    return [int(x1), int(y1), int(x2), int(y2)]

def draw_points_on_image(image, points):
    """Draws red dots on the image to indicate click locations."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    for pt in points:
        x, y = pt
        r = 8 # Radius of point
        draw.ellipse((x-r, y-r, x+r, y+r), fill="red", outline="white", width=4)
    
    return draw_img
