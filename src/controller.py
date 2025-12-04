from .schemas import GlobalStore, ObjectState, SelectorInput
from .inference import search_objects, refine_object
from PIL import Image
import numpy as np

class AppController:
    def __init__(self):
        self.store = GlobalStore()
        self.current_image = None # PIL Image
        
    def set_image(self, image: Image.Image):
        # Check if image is different before resetting store
        if self.current_image is not None and image is not None:
            if self.current_image.size == image.size:
                # Simple check: if size matches, check a few pixels or full array
                # Converting to numpy for full check is safest to avoid stale masks on new image
                if np.array_equal(np.array(self.current_image), np.array(image)):
                    return # Same image, keep store
        
        self.current_image = image
        self.store = GlobalStore() # Reset store on new image
        
    def search_and_add(self, class_name: str, search_boxes: list[list[int]] = [], search_labels: list[int] = [], class_name_override: str = None):
        if self.current_image is None: return []
        
        # Create SelectorInput
        selector_input = SelectorInput(
            image=self.current_image,
            text=class_name,
            class_name_override=class_name_override,
            input_boxes=search_boxes,
            input_labels=search_labels
        )
        
        candidates = search_objects(selector_input)
        
        # We return candidates, but don't add to store yet (UI will decide)
        return candidates

    def add_candidates_to_store(self, candidates: list[ObjectState], selected_indices: list[int]):
        added_ids = []
        for idx in selected_indices:
            if 0 <= idx < len(candidates):
                obj_state = candidates[idx]
                self.store.objects[obj_state.object_id] = obj_state
                added_ids.append(obj_state.object_id)
        return added_ids
        
    def refine_object(self, obj_id: str, point: list[int], label: int):
        if obj_id not in self.store.objects: return None
        if self.current_image is None: return None
        
        obj = self.store.objects[obj_id]
        
        # Update history
        obj.input_points.append(point)
        obj.input_labels.append(label)
        
        # Run Refiner
        new_mask = refine_object(self.current_image, obj)
        
        # Update Mask
        obj.binary_mask = new_mask
        
        return new_mask

    def undo_last_point(self, obj_id: str):
        if obj_id not in self.store.objects: return None
        obj = self.store.objects[obj_id]
        
        if not obj.input_points:
            return obj.binary_mask # Nothing to undo
            
        # Remove last
        obj.input_points.pop()
        obj.input_labels.pop()
        
        # If no points left, revert to initial
        if not obj.input_points:
            obj.binary_mask = obj.initial_mask
            return obj.binary_mask
            
        # Otherwise re-run refinement
        new_mask = refine_object(self.current_image, obj)
        obj.binary_mask = new_mask
        return new_mask

    def remove_object(self, obj_id: str):
        if obj_id in self.store.objects:
            del self.store.objects[obj_id]
            return True
        return False

    def revert_object(self, obj_id: str):
        """Revert object to its initial state (before refinement)."""
        if obj_id not in self.store.objects: return None
        obj = self.store.objects[obj_id]
        
        # Reset to initial mask
        obj.binary_mask = obj.initial_mask
        # Clear points
        obj.input_points = []
        obj.input_labels = []
        
        return obj.binary_mask

    def export_data(self, output_dir: str):
        """Export image and annotations to output_dir."""
        import os
        import uuid
        
        if self.current_image is None: return None
        if not self.store.objects: return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique name
        base_name = f"export_{uuid.uuid4().hex[:8]}"
        # img_path = os.path.join(output_dir, f"{base_name}.jpg")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # Save Image - Skipped as per user request
        # if self.current_image.mode == "RGBA":
        #     self.current_image.convert("RGB").save(img_path)
        # else:
        #     self.current_image.save(img_path)
        
        # Save Annotations (YOLO Segmentation Format)
        # class_id x1 y1 x2 y2 ... (normalized)
        # We don't have class IDs mapped to integers globally, so we'll use 0 for now 
        # or map unique class names to IDs.
        
        class_map = {}
        next_id = 0
        
        lines = []
        w, h = self.current_image.size
        
        for obj in self.store.objects.values():
            if obj.class_name not in class_map:
                class_map[obj.class_name] = next_id
                next_id += 1
            
            cid = class_map[obj.class_name]
            
            # Get contours from mask
            import cv2
            mask = obj.binary_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # Flatten and normalize
                points = cnt.flatten()
                norm_points = []
                for i in range(0, len(points), 2):
                    nx = points[i] / w
                    ny = points[i+1] / h
                    norm_points.extend([f"{nx:.6f}", f"{ny:.6f}"])
                
                if len(norm_points) > 4: # At least 3 points (6 coords) needed for a polygon really
                    line = f"{cid} " + " ".join(norm_points)
                    lines.append(line)
                    
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
            
        # Also save class map
        with open(os.path.join(output_dir, "classes.txt"), "w") as f:
            for name, cid in class_map.items():
                f.write(f"{name}\n")
                
        return None, txt_path

    def get_all_masks(self):
        return [(obj.binary_mask, f"{obj.class_name}") for obj in self.store.objects.values()]
        
    def get_object_mask(self, obj_id):
        if obj_id in self.store.objects:
            return self.store.objects[obj_id].binary_mask
        return None

# Global Controller
controller = AppController()
