from .schemas import GlobalStore, ObjectState, SelectorInput, ProjectState
from .inference import search_objects, refine_object
from .dataset_manager import DatasetManager
from .view_helpers import draw_candidates
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
        self.active_project_path = None # Path to the current project JSON file
        
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

        
    def reset_project(self):
        """Reset the project state completely."""
        self.store = GlobalStore()
        self.current_image = None
        self.current_image_path = None
        self.project = ProjectState()
        self.global_class_map = {}
        self.active_project_path = None

    def auto_save(self):
        """Auto-save the project if an active path is set."""
        if self.active_project_path:
            print(f"💾 Auto-saving to {self.active_project_path}...")
            return self.save_project(self.active_project_path)
        return False, "No active project to save."

    def update_history(self, prompt: str, class_name: str):
        if prompt and prompt not in self.project.prompt_history:
            self.project.prompt_history.append(prompt)
        if class_name and class_name not in self.project.class_name_history:
            self.project.class_name_history.append(class_name)

    def update_history(self, prompt: str, class_name: str):
        if prompt and prompt not in self.project.prompt_history:
            self.project.prompt_history.append(prompt)
        if class_name and class_name not in self.project.class_name_history:
            self.project.class_name_history.append(class_name)

    def search_and_add(self, class_name: str, search_boxes: list[list[int]] = [], search_labels: list[int] = [], class_name_override: str = None, crop_box: list[int] = None):
        self.update_history(class_name, class_name_override)
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

    def get_candidate_preview(self, candidates: list[ObjectState], selected_index: int | set | list = None):
        """Generate preview image with candidates drawn."""
        if self.current_image is None or not candidates:
            return self.current_image
            
        return draw_candidates(self.current_image, candidates, selected_index)

    def get_candidates_dataframe(self, candidates: list[ObjectState]):
        """Get dataframe for UI list."""
        data = []
        for i, obj in enumerate(candidates):
            # Add ID column (i+1) to match the image labels
            data.append([
                i + 1,              # ID
                obj.class_name,     # Class
                f"{obj.score:.2f}"  # Score
            ])
        return data
        
    def refine_object(self, obj_id: str, point: list[int], label: int):
        if obj_id not in self.store.objects: return None
        if self.current_image is None: return None
        
        obj = self.store.objects[obj_id]
        
        # Update history
        obj.input_points.append(point)
        obj.input_labels.append(label)
        
        print(f"Refining {obj_id}: Points={obj.input_points}, Labels={obj.input_labels}")
        
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
        print(f"Refining (Undo) {obj_id}: Points={obj.input_points}, Labels={obj.input_labels}")
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

    def export_data(self, output_dir: str, purge: bool = False, zip_output: bool = False):
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
        
        if purge:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

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
"""
        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            f.write(yaml_content)
            
        msg = f"Exported {exported_count} images to {output_dir}"
        
        if zip_output:
            # Determine zip name based on project name if available
            zip_name = "dataset"
            if self.active_project_path:
                # Extract project name from path (e.g., "saved_projects/my_project.json" -> "my_project")
                zip_name = os.path.splitext(os.path.basename(self.active_project_path))[0]
            
            # Create a temp folder for staging the zip
            parent_dir = os.path.dirname(os.path.abspath(output_dir))
            temp_dir = os.path.join(parent_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.join(temp_dir, zip_name)
            
            # Create zip in temp folder
            zip_file = shutil.make_archive(base_name, 'zip', output_dir)
            
            # Clear output_dir
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            # Move zip to output_dir
            final_name = f"{zip_name}.zip"
            final_path = os.path.join(output_dir, final_name)
            shutil.move(zip_file, final_path)
            
            # Remove temp folder
            shutil.rmtree(temp_dir)
            
            msg += f" and zipped to {final_name} (original files deleted)"
            
        return None, msg

    def save_project(self, file_path: str):
        """Save project state to JSON and bundle images."""
        import json
        import os
        import shutil
        from .utils import mask_to_polygons
        
        # Ensure current state is saved
        if self.current_image_path:
            self.project.annotations[self.current_image_path] = self.store
            
        # Create assets directory
        base_dir = os.path.dirname(file_path)
        project_name = os.path.splitext(os.path.basename(file_path))[0]
        assets_dir_name = f"{project_name}_assets"
        assets_dir = os.path.join(base_dir, assets_dir_name)
        os.makedirs(assets_dir, exist_ok=True)
        
        # Map original paths to relative paths
        path_map = {} # original -> relative
        new_playlist = []
        
        # Process playlist
        for original_path in self.project.playlist:
            filename = os.path.basename(original_path)
            # Handle duplicate filenames by prepending index if needed? 
            # For now assume unique filenames or just overwrite (simple)
            # Better: check collision
            
            dest_path = os.path.join(assets_dir, filename)
            
            # Copy file if it doesn't exist or if we want to ensure it's there
            try:
                if not os.path.exists(dest_path) or os.path.abspath(original_path) != os.path.abspath(dest_path):
                    shutil.copy2(original_path, dest_path)
            except Exception as e:
                print(f"Warning: Failed to copy {original_path} to {dest_path}: {e}")
            
            # Store relative path
            relative_path = os.path.join(assets_dir_name, filename)
            path_map[original_path] = relative_path
            new_playlist.append(relative_path)
            
        data = {
            "playlist": new_playlist,
            "current_index": self.project.current_index,
            "prompt_history": self.project.prompt_history,
            "class_name_history": self.project.class_name_history,
            "annotations": {}
        }
        
        for path, store in self.project.annotations.items():
            # Get the new relative path key
            new_key = path_map.get(path)
            if not new_key:
                # If annotation exists for a file not in playlist (shouldn't happen but safe fallback)
                filename = os.path.basename(path)
                new_key = os.path.join(assets_dir_name, filename)
                
            objects_data = {}
            for obj_id, obj in store.objects.items():
                objects_data[obj_id] = {
                    "object_id": obj.object_id,
                    "score": obj.score,
                    "class_name": obj.class_name,
                    "anchor_box": obj.anchor_box,
                    "input_points": obj.input_points,
                    "input_labels": obj.input_labels,
                    "polygons": mask_to_polygons(obj.binary_mask)
                }
            data["annotations"][new_key] = objects_data
            
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update active project path
            self.active_project_path = file_path
            
            return True, f"Project saved to {file_path} (Images bundled in {assets_dir_name})"
        except Exception as e:
            return False, f"Failed to save project: {e}"

    def load_project(self, file_path: str):
        """Load project state from JSON."""
        import json
        import os
        from .utils import polygons_to_mask
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return False, f"Failed to load file: {e}"
            
        base_dir = os.path.dirname(file_path)
        
        # Reconstruct absolute paths for playlist
        loaded_playlist = []
        for rel_path in data.get("playlist", []):
            abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
            loaded_playlist.append(abs_path)
            
        # Restore Project State
        self.project = ProjectState(
            playlist=loaded_playlist,
            current_index=data.get("current_index", -1),
            prompt_history=data.get("prompt_history", []),
            class_name_history=data.get("class_name_history", [])
        )
        
        # Restore Annotations
        missing_files = []
        for rel_path, objects_data in data.get("annotations", {}).items():
            abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
            store = GlobalStore(image_path=abs_path)
            
            # Need image size to restore masks
            try:
                with Image.open(abs_path) as img:
                    w, h = img.size
            except:
                print(f"Warning: Could not read image {abs_path} during load. Skipping masks.")
                missing_files.append(abs_path)
                continue
                
            for obj_id, obj_data in objects_data.items():
                # Reconstruct mask
                polygons = obj_data.get("polygons", [])
                mask = polygons_to_mask(polygons, w, h)
                
                obj = ObjectState(
                    object_id=obj_data["object_id"],
                    score=obj_data["score"],
                    class_name=obj_data["class_name"],
                    anchor_box=obj_data["anchor_box"],
                    binary_mask=mask,
                    initial_mask=mask.copy(), # Assume loaded state is initial
                    input_points=obj_data.get("input_points", []),
                    input_labels=obj_data.get("input_labels", [])
                )
                store.objects[obj_id] = obj
                
            self.project.annotations[abs_path] = store
            
        # Load current image
        if self.project.current_index >= 0:
            self.load_image_at_index(self.project.current_index)
            
        msg = f"Project loaded from {file_path}"
        if missing_files:
            msg += f". Warning: {len(missing_files)} images not found (annotations skipped)."
            
        # Update active project path
        self.active_project_path = file_path
            
        return True, msg

    def get_all_masks(self):
        return [(obj.binary_mask, f"{obj.class_name}") for obj in self.store.objects.values()]
        
    def get_object_mask(self, obj_id):
        if obj_id in self.store.objects:
            return self.store.objects[obj_id].binary_mask
        return None

    def clean_and_export_dataset(self, dataset_path, tolerance_ratio=0.000805, min_area_ratio=0.000219):
        """Clean, validate, and zip a YOLO dataset."""
        manager = DatasetManager(dataset_path)
        
        # 1. Remove Zone.Identifier files
        manager.remove_zone_identifiers()
        
        # 2. Clean dataset (in-place)
        print(f"Cleaning dataset at {dataset_path}...")
        stats = manager.cleanup_dataset(tolerance_ratio, min_area_ratio)
        
        # 3. Finalize (Validation folders + Zip)
        print("Finalizing dataset...")
        zip_path = manager.finalize_dataset(create_zip=True)
        
        return stats, zip_path

# Global Controller
controller = AppController()
