
import os
import sys
import shutil
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
from PIL import Image, ImageDraw
from src.sam3_annotation_tool.controller import controller
from src.sam3_annotation_tool.utils import apply_mask_overlay, draw_points_on_image
from src.sam3_annotation_tool.dataset_manager import DatasetManager

OUTPUT_DIR = "test_output"

def draw_box(image, box, color="blue", width=5):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=color, width=width)
    return image

def print_store_status(controller, step_name):
    store = controller.store
    print(f"\n📊 --- Global Store Status [{step_name}] ---")
    print(f"   Total Objects: {len(store.objects)}")
    
    for obj_id, obj in store.objects.items():
        mask_pixels = np.count_nonzero(obj.binary_mask) if obj.binary_mask is not None else 0
        print(f"   🔹 Object ID: {obj_id}")
        print(f"      Class: {obj.class_name}, Score: {obj.score:.2f}")
        print(f"      Anchor Box: {obj.anchor_box}")
        print(f"      Mask Non-Zero Pixels: {mask_pixels}")
        print(f"      Refinement Points: {len(obj.input_points)}")
    print("------------------------------------------\n")

def test_controller_flow():
    print("🚀 Starting Controller Flow Test...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Setup Image
    # Dynamically find an image in example_img
    example_dir = os.path.join(os.getcwd(), "example_img")
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    try:
        img_filename = next(f for f in os.listdir(example_dir) if f.lower().endswith(valid_extensions))
        img_path = os.path.join(example_dir, img_filename)
        print(f"Using test image: {img_path}")
    except (StopIteration, FileNotFoundError):
        print(f"❌ No images found in {example_dir}")
        return
        
    if not os.path.exists(img_path):
        print(f"❌ Image not found at {img_path}")
        return
        
    # Create a second dummy image for playlist testing
    img_path_2 = os.path.join(OUTPUT_DIR, "test_image_2.png")
    shutil.copy2(img_path, img_path_2)
    
    image = Image.open(img_path)
    print(f"✅ Loaded image from {img_path} (Size: {image.size})")
    
    # 2. Initialize Controller
    print("🔹 Loading playlist in controller...")
    # Use load_playlist to properly set up project state for export
    controller.load_playlist([img_path, img_path_2])
    
    # Verify image loaded
    if controller.current_image is None:
        print("❌ Failed to load image via load_playlist")
        return
    
    # 3. Define Inputs
    prompt = "yellow cheese blocks" 
    
    # Save Input Visualization
    vis_input = image.copy()
    vis_input.save(os.path.join(OUTPUT_DIR, "01_search_input.png"))
    print(f"💾 Saved search input visualization to {OUTPUT_DIR}/01_search_input.png")
    
    print(f"🔹 Running Search with prompt='{prompt}' (No Box)...")
    
    # 4. Run Search
    try:
        candidates = controller.search_and_add(
            class_name=prompt,
            search_boxes=[],
            search_labels=[]
        )
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return

    print(f"✅ Search complete. Found {len(candidates)} candidates.")
    
    if len(candidates) == 0:
        print("⚠️ No candidates found. Test cannot proceed to refinement.")
        return

    # Visualize Candidates
    # Create a composite of all masks
    if candidates:
        all_masks = np.array([c.binary_mask for c in candidates])
        vis_results = apply_mask_overlay(image.copy(), all_masks, opacity=0.5)
        vis_results.save(os.path.join(OUTPUT_DIR, "02_search_results.png"))
        print(f"💾 Saved search results visualization to {OUTPUT_DIR}/02_search_results.png")

    # 5. Add to Store
    print(f"🔹 Adding all {len(candidates)} candidates to store...")
    
    print_store_status(controller, "Before Adding")
    
    # Select all candidates
    all_indices = list(range(len(candidates)))
    added_ids = controller.add_candidates_to_store(candidates, all_indices)
    
    if not added_ids:
        print("❌ Failed to add candidates to store.")
        return
        
    print(f"✅ Added {len(added_ids)} objects to store.")
    
    print_store_status(controller, "After Adding")
    
    # 6. Test Refinement with Random Points
    # We will refine the first object in the list
    obj_id = added_ids[0]
    print(f"🔹 Testing Refinement on first object (ID: {obj_id})...")
    print("🔹 Testing Refinement with random points...")
    
    # We will accumulate points to draw them all
    points_history = []
    w, h = image.size
    
    for i in range(3):
        # Generate random point within image
        rx = random.randint(0, w-1)
        ry = random.randint(0, h-1)
        click_point = [rx, ry]
        click_label = 1 # Positive click
        points_history.append(click_point)
        
        print(f"   Refining with point {click_point}...")
        
        try:
            new_mask = controller.refine_object(obj_id, click_point, click_label)
            if new_mask is not None:
                print(f"   ✅ Refinement {i+1} successful. Mask shape: {new_mask.shape}, Non-zero pixels: {np.count_nonzero(new_mask)}")
                
                # Visualize Refinement
                # Overlay the NEW mask on the image
                vis_refine = apply_mask_overlay(image.copy(), new_mask, opacity=0.6)
                # Draw the points
                vis_refine = draw_points_on_image(vis_refine, points_history)
                
                filename = f"03_refinement_step_{i+1}.png"
                vis_refine.save(os.path.join(OUTPUT_DIR, filename))
                print(f"   💾 Saved refinement step {i+1} to {OUTPUT_DIR}/{filename}")
                
                print_store_status(controller, f"After Refinement {i+1}")
                
            else:
                print(f"   ❌ Refinement {i+1} returned None.")
        except Exception as e:
            print(f"   ❌ Refinement {i+1} failed: {e}")

    # 7. Test Revert Object
    print(f"🔹 Testing Revert Object (ID: {obj_id})...")
    try:
        reverted_mask = controller.revert_object(obj_id)
        if reverted_mask is not None:
            print(f"   ✅ Revert successful. Mask shape: {reverted_mask.shape}, Non-zero pixels: {np.count_nonzero(reverted_mask)}")
            print_store_status(controller, "After Revert")
        else:
            print("   ❌ Revert returned None.")
    except Exception as e:
        print(f"   ❌ Revert failed: {e}")

    # 7.5 Test Next Image (Simulate Finish Image)
    print("🔹 Testing Next Image (Simulate Finish Image)...")
    print(f"   Current Index: {controller.project.current_index}")
    print(f"   Current Path: {controller.current_image_path}")
    
    # Simulate what app.py does: save explicitly then next
    if controller.current_image_path:
        controller.project.annotations[controller.current_image_path] = controller.store
        
    next_img = controller.next_image()
    
    if next_img:
        print("   ✅ Next image loaded.")
        print(f"   New Index: {controller.project.current_index}")
        print(f"   New Path: {controller.current_image_path}")
        
        # Check if previous image is in annotations
        prev_path = img_path
        if prev_path in controller.project.annotations:
            saved_store = controller.project.annotations[prev_path]
            print(f"   ✅ Previous image found in annotations.")
            print(f"      Objects in saved store: {len(saved_store.objects)}")
            if len(saved_store.objects) > 0:
                print("      ✅ Saved store has objects.")
            else:
                print("      ❌ Saved store is empty!")
        else:
            print(f"   ❌ Previous image NOT found in annotations. Keys: {list(controller.project.annotations.keys())}")
    else:
        print("   ❌ Failed to load next image.")

    # 8. Test Export Data
    print("🔹 Testing Export Data...")
    try:
        _, msg = controller.export_data(OUTPUT_DIR)
        print(f"   ℹ️ Export Message: {msg}")
        
        if "No annotations" in msg:
             print("   ❌ Export failed: No annotations found.")
        else:
            print(f"   ✅ Export reported success.")
            
            # Use the robust verification script
            print("\n   🔍 Running robust dataset verification...")
            manager = DatasetManager(OUTPUT_DIR)
            is_valid = manager.verify_dataset()
            if is_valid:
                print("   ✅ Dataset verification PASSED.")
            else:
                print("   ❌ Dataset verification FAILED.")
                
            # Simple check for specific file existence (as a double check)
            # Construct expected label path
            # Image name: DEPAL1_... .png -> Label: DEPAL1_... .txt
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            expected_label_path = os.path.join(OUTPUT_DIR, "labels", "train", label_filename)
            
            print(f"      Checking for: {expected_label_path}")
            
            # Verify file existence
            if os.path.exists(expected_label_path):
                print("      ✅ Label file exists on disk.")
                
                # Verify content
                with open(expected_label_path, "r") as f:
                    content = f.read()
                    print(f"      Annotation Content Preview:\n{content[:200]}...")
                    if len(content.strip()) > 0:
                        print("      ✅ Annotation file is not empty.")
                    else:
                        print("      ❌ Annotation file is empty!")
            else:
                print("      ❌ Label file missing on disk.")
                
            # Check data.yaml
            yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
            if os.path.exists(yaml_path):
                print("      ✅ data.yaml exists.")
            else:
                print("      ❌ data.yaml missing.")

    except Exception as e:
        print(f"   ❌ Export failed: {e}")

    print("🎉 Test Complete!")

if __name__ == "__main__":
    test_controller_flow()
