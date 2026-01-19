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

def mask_to_polygons(mask: np.ndarray) -> list[list[int]]:
    """Convert binary mask to list of polygons (flattened [x, y, x, y...])."""
    import cv2
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        points = cnt.flatten().tolist()
        if len(points) >= 6: # At least 3 points
            polygons.append(points)
    return polygons

def polygons_to_mask(polygons: list[list[int]], width: int, height: int) -> np.ndarray:
    """Convert list of polygons back to binary mask."""
    import cv2
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def clean_polygon_shapely(normalized_coords, img_width, img_height, 
                          tolerance_ratio=0.0012, min_area_ratio=0.00001):
    """Clean polygon using shapely with resolution-scaled parameters.
    
    This function applies topology validation, repair, simplification, and area filtering
    to polygon coordinates. It's designed to work with YOLO-format normalized coordinates.
    
    The Douglas-Peucker simplification algorithm removes points that are closer than
    tolerance_px to the simplified line. Simple polygons with widely-spaced points are
    naturally preserved, while complex polygons with many close points are simplified.
    
    Args:
        normalized_coords: List of [x, y, x, y, ...] in range [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
        tolerance_ratio: Simplification tolerance as fraction of min(width, height).
                        Default 0.0012 means ~1.5px on typical resolutions.
                        Higher values = more aggressive simplification.
        min_area_ratio: Minimum polygon area as fraction of total image area.
                       Polygons smaller than this are filtered out.
                       Default 0.00001 = 0.001% of image area (~27px² at 2208x1242).
    
    Returns:
        Tuple of (cleaned_normalized_coords, stats_dict) or (None, stats_dict) if filtered.
        
        cleaned_normalized_coords: List of [x, y, x, y, ...] in range [0, 1], or None
        
        stats_dict contains:
            - original_points (int): Number of points before cleanup
            - final_points (int): Number of points after cleanup
            - original_area (float): Area in square pixels before cleanup
            - final_area (float): Area in square pixels after cleanup
            - was_invalid (bool): Whether polygon had topology issues (self-intersection, etc.)
            - was_filtered (bool): Whether polygon was removed
            - filter_reason (str or None): Reason for filtering if was_filtered=True
    
    Example:
        >>> # Typical usage with YOLO coordinates
        >>> normalized = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 3 points
        >>> cleaned, stats = clean_polygon_shapely(normalized, 1920, 1080)
        >>> if cleaned is not None:
        >>>     print(f"Reduced from {stats['original_points']} to {stats['final_points']} points")
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        # Fallback: return original without cleanup if shapely not available
        return normalized_coords, {
            'original_points': len(normalized_coords) // 2,
            'final_points': len(normalized_coords) // 2,
            'original_area': 0.0,
            'final_area': 0.0,
            'was_invalid': False,
            'was_filtered': False,
            'filter_reason': 'shapely_not_installed'
        }
    
    stats = {
        'original_points': len(normalized_coords) // 2,
        'final_points': 0,
        'original_area': 0.0,
        'final_area': 0.0,
        'was_invalid': False,
        'was_filtered': False,
        'filter_reason': None
    }
    
    # Need at least 3 points for a valid polygon
    if len(normalized_coords) < 6:
        stats['was_filtered'] = True
        stats['filter_reason'] = 'too_few_points'
        return None, stats
    
    # Denormalize to pixel coordinates
    pixel_coords = []
    for i in range(0, len(normalized_coords), 2):
        pixel_coords.append(normalized_coords[i] * img_width)
        pixel_coords.append(normalized_coords[i + 1] * img_height)
    
    # Calculate original area using shoelace formula
    def shoelace_area(coords):
        if len(coords) < 6:
            return 0.0
        x = coords[::2]
        y = coords[1::2]
        area = 0.0
        n = len(x)
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]
        return abs(area) / 2.0
    
    original_area = shoelace_area(pixel_coords)
    stats['original_area'] = original_area
    
    # Filter by minimum area
    min_area_px = min_area_ratio * img_width * img_height
    if original_area < min_area_px:
        stats['was_filtered'] = True
        stats['filter_reason'] = f'area_too_small_{original_area:.1f}px2'
        return None, stats
    
    try:
        # Convert to shapely polygon (list of (x, y) tuples)
        points = [(pixel_coords[i], pixel_coords[i+1]) 
                  for i in range(0, len(pixel_coords), 2)]
        
        poly = ShapelyPolygon(points)
        
        # Check and repair validity (handles self-intersections, etc.)
        if not poly.is_valid:
            stats['was_invalid'] = True
            poly = poly.buffer(0)  # Auto-repair using buffer trick
            
            # If still invalid or empty after repair, filter out
            if not poly.is_valid or poly.is_empty:
                stats['was_filtered'] = True
                stats['filter_reason'] = 'invalid_unfixable'
                return None, stats
        
        # Calculate tolerance in pixels
        # This is the maximum distance a point can be from the simplified line
        # to be removed. Simple polygons naturally won't be affected.
        tolerance_px = tolerance_ratio * min(img_width, img_height)
        
        # Apply Douglas-Peucker simplification
        # preserve_topology=True prevents creating self-intersections
        simplified = poly.simplify(tolerance_px, preserve_topology=True)
        
        # Check if simplification resulted in invalid geometry
        if simplified.is_empty or not simplified.is_valid:
            stats['was_filtered'] = True
            stats['filter_reason'] = 'simplification_failed'
            return None, stats
        
        # Extract coordinates from simplified polygon
        if hasattr(simplified, 'exterior'):
            # shapely polygons have exterior coordinates
            # coords includes duplicate last point, so exclude it
            coords = list(simplified.exterior.coords[:-1])
        elif hasattr(simplified, 'geoms'):
            # Handle MultiPolygon: take the largest polygon by area
            largest_poly = max(simplified.geoms, key=lambda p: p.area)
            coords = list(largest_poly.exterior.coords[:-1])
        else:
            # Shouldn't happen, but handle gracefully
            stats['was_filtered'] = True
            stats['filter_reason'] = 'no_exterior'
            return None, stats
        
        # Need at least 3 points after simplification
        if len(coords) < 3:
            stats['was_filtered'] = True
            stats['filter_reason'] = 'simplified_too_few_points'
            return None, stats
        
        # Convert back to flat list [x, y, x, y, ...]
        cleaned_pixel_coords = []
        for x, y in coords:
            cleaned_pixel_coords.extend([x, y])
        
        # Calculate final area
        final_area = shoelace_area(cleaned_pixel_coords)
        stats['final_area'] = final_area
        stats['final_points'] = len(coords)
        
        # Normalize back to [0, 1] range for YOLO format
        cleaned_normalized = []
        for i in range(0, len(cleaned_pixel_coords), 2):
            cleaned_normalized.append(cleaned_pixel_coords[i] / img_width)
            cleaned_normalized.append(cleaned_pixel_coords[i + 1] / img_height)
        
        return cleaned_normalized, stats
        
    except Exception as e:
        # Catch any unexpected errors and filter out problematic polygons
        stats['was_filtered'] = True
        stats['filter_reason'] = f'exception_{str(e)[:50]}'
        return None, stats
