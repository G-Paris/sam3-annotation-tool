from .schemas import GlobalStore, ObjectState, SelectorInput, ProjectState
from .inference import search_objects, refine_object
from PIL import Image
import numpy as np
import os
import shutil
import uuid
import cv2

class AppController:
    def __init__(self):
        self.store = GlobalStore()
        self.current_image = None # PIL Image
        self.current_image_path = None # Path to current image
        
        # Playlist state
        self.project = ProjectState()
        self.global_class_map = {} # Map class_name -> int ID
        
    def load_playlist(self, file_paths: list[str]):
        """Load a list of image paths."""
        # Filter for images
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        playlist = sorted([p for p in file_paths if os.path.splitext(p)[1].lower() in valid_exts])
        
        self.project = ProjectState(playlist=playlist)
        self.current_image = None
        self.current_image_path = None
        self.store = GlobalStore()
        
        if self.project.playlist:
            return self.load_image_at_index(0)
        return None

    def load_image_at_index(self, index: int):
        if not self.project.playlist or index < 0 or index >= len(self.project.playlist):
            return None
            
        # Save current state if we have an image loaded
        if self.current_image_path:
            self.project.annotations[self.current_image_path] = self.store
            
        self.project.current_index = index
        path = self.project.playlist[index]
        
        try:
            image = Image.open(path).convert("RGB")
            self.current_image = image
            self.current_image_path = path
            
            # Restore store if exists, else new
            if path in self.project.annotations:
                self.store = self.project.annotations[path]
            else:
                self.store = GlobalStore(image_path=path)
                
            return image
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def next_image(self):
        return self.load_image_at_index(self.project.current_index + 1)

    def prev_image(self):
        return self.load_image_at_index(self.project.current_index - 1)
        
    def set_image(self, image: Image.Image):
        # Legacy support: treat as single image playlist without path
        # This might break if we rely on paths for export. 
        # Ideally we force file upload.
        # For now, let's just set it and reset store, but warn it won't work well with playlist export
        self.current_image = image
        self.current_image_path = None
        self.store = GlobalStore()
        self.project = ProjectState()

        
    def search_and_add(self, class_name: str, search_boxes: list[list[int]] = [], search_labels: list[int] = [], class_name_override: str = None, crop_box: list[int] = None):
        if self.current_image is None: return []
        
        # Create SelectorInput
        selector_input = SelectorInput(
            image=self.current_image,
            text=class_name,
            class_name_override=class_name_override,
            input_boxes=search_boxes,
            input_labels=search_labels,
            crop_box=crop_box
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
        """Export all images and annotations in playlist to YOLO format."""
        
        # Ensure current state is saved
        if self.current_image_path:
            self.project.annotations[self.current_image_path] = self.store
            
        if not self.project.annotations:
            return None, "No annotations to export."
            
        # Structure:
        # output_dir/
        #   data.yaml
        #   images/
        #     train/
        #   labels/
        #     train/
        
        images_dir = os.path.join(output_dir, "images", "train")
        labels_dir = os.path.join(output_dir, "labels", "train")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Collect all unique class names to build map
        all_class_names = set()
        for store in self.project.annotations.values():
            for obj in store.objects.values():
                all_class_names.add(obj.class_name)
        
        # Update global map (append new ones)
        sorted_classes = sorted(list(all_class_names))
        class_list = sorted_classes
        class_map = {name: i for i, name in enumerate(class_list)}
        
        exported_count = 0
        
        for path, store in self.project.annotations.items():
            if not store.objects:
                continue
                
            # Copy image
            filename = os.path.basename(path)
            dest_img_path = os.path.join(images_dir, filename)
            shutil.copy2(path, dest_img_path)
            
            # Generate Label File
            label_filename = os.path.splitext(filename)[0] + ".txt"
            dest_label_path = os.path.join(labels_dir, label_filename)
            
            # We need image size for normalization. 
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except:
                print(f"Could not read image size for {path}")
                continue
                
            lines = []
            for obj in store.objects.values():
                cid = class_map.get(obj.class_name, 0)
                
                mask = obj.binary_mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    points = cnt.flatten()
                    if len(points) < 6: continue # Need at least 3 points
                    
                    norm_points = []
                    for i in range(0, len(points), 2):
                        nx = points[i] / w
                        ny = points[i+1] / h
                        # Clip to 0-1
                        nx = max(0, min(1, nx))
                        ny = max(0, min(1, ny))
                        norm_points.extend([f"{nx:.6f}", f"{ny:.6f}"])
                        
                    line = f"{cid} " + " ".join(norm_points)
                    lines.append(line)
            
            with open(dest_label_path, "w") as f:
                f.write("\n".join(lines))
                
            exported_count += 1
            
        # Create data.yaml
        yaml_content = f"""names:
{chr(10).join([f"  {i}: {name}" for i, name in enumerate(class_list)])}
path: .
train: images/train
val: images/train
"""
        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            f.write(yaml_content)
            
        return None, f"Exported {exported_count} images to {output_dir}"

    def get_all_masks(self):
        return [(obj.binary_mask, f"{obj.class_name}") for obj in self.store.objects.values()]
        
    def get_object_mask(self, obj_id):
        if obj_id in self.store.objects:
            return self.store.objects[obj_id].binary_mask
        return None

# Global Controller
controller = AppController()
