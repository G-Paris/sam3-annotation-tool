import os
import sys
import gradio as gr
from PIL import Image
import numpy as np

# Add current directory to path so we can import app
sys.path.append(os.getcwd())

from app import run_inference_step1, render_results_step2, add_to_store, init_editor, select_all_candidates, revert_object_refinement, export_results
from src.sam3_annotation_tool.controller import controller

# Mock Gradio Update for inspection if needed, 
# but we are importing real gradio so we will get real Update objects.

def test_ui_flow():
    print("🚀 Starting UI Callback Test...")
    
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
        
    image = Image.open(img_path)
    print(f"✅ Loaded image: {image.size}")
    
    # Mock ImageEditor input
    image_input = {
        "background": image,
        "composite": image,
        "layers": []
    }
    
    # 2. Test run_inference
    print("\n🔹 Testing run_inference...")
    prompt = "yellow cheese blocks"
    boxes = []
    labels = []
    
    try:
        # Step 1
        candidates, current_image, input_vis, result_vis = run_inference_step1(
            image_input, prompt, boxes, labels, None
        )
        print(f"✅ Step 1 returned successfully.")
        print(f"   Candidates found: {len(candidates)}")
        
        # Step 2
        gallery_items, preview_img, selector_update = render_results_step2(candidates, current_image)
        print(f"✅ Step 2 returned successfully.")
        
        # Handle Gallery Update
        if isinstance(gallery_items, dict) and 'value' in gallery_items:
            print(f"   Gallery items count: {len(gallery_items['value'])}")
        else:
            print(f"   Gallery items: {gallery_items}")
            
        # Handle Preview Image Update
        os.makedirs("test_output", exist_ok=True)
        if isinstance(preview_img, dict) and 'value' in preview_img:
            real_img = preview_img['value']
            if real_img:
                real_img.save("test_output/05_ui_preview_overlay.png") # type: ignore
                print(f"   💾 Saved preview overlay to test_output/05_ui_preview_overlay.png")
            else:
                print("   ❌ Preview image value is None")
        elif hasattr(preview_img, 'save'):
             preview_img.save("test_output/05_ui_preview_overlay.png") # type: ignore
             print(f"   💾 Saved preview overlay to test_output/05_ui_preview_overlay.png")
        else:
            print(f"   ❌ Preview image is unexpected type: {type(preview_img)}")

        print(f"   Selected Indices State: {selector_update}")
        
    except Exception as e:
        print(f"❌ run_inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2b. Test Select All
    print("\n🔹 Testing select_all_candidates...")
    selected_indices = select_all_candidates(candidates)
    print(f"   Selected indices count: {len(selected_indices)}")
    if len(selected_indices) != len(candidates):
        print("   ❌ Select all failed to return all indices.")

    # 3. Test add_to_store
    print("\n🔹 Testing add_to_store...")
    if len(candidates) > 0:
        # Select the first one
        selected_indices = [0] 
        
        try:
            msg, res_vis, ed_vis = add_to_store(candidates, selected_indices)
            print(f"✅ add_to_store returned: {msg}")
            
            # Verify it's in the controller
            print(f"   Controller store count: {len(controller.store.objects)}")
            
        except Exception as e:
            print(f"❌ add_to_store failed: {e}")
            return
    else:
        print("⚠️ No candidates to add.")

    # 4. Test init_editor
    print("\n🔹 Testing init_editor...")
    try:
        overlay_img, radio_update = init_editor()
        
        if overlay_img:
            print(f"✅ init_editor returned overlay image: {overlay_img.size}")
            # Save it to verify
            overlay_img.save("test_output/04_ui_editor_overlay.png")
            print(f"   Saved overlay to test_output/04_ui_editor_overlay.png")
        else:
            print("❌ init_editor returned None image")
            
        print(f"   Radio update: {radio_update}")
        if isinstance(radio_update, dict) and 'choices' in radio_update:
             print(f"   Radio choices: {radio_update['choices']}")
        elif hasattr(radio_update, 'kwargs') and 'choices' in radio_update.kwargs: # type: ignore
             print(f"   Radio choices: {radio_update.kwargs['choices']}") # type: ignore
             
    except Exception as e:
        print(f"❌ init_editor failed: {e}")

    # 5. Test Revert (UI Wrapper)
    print("\n🔹 Testing revert_object_refinement...")
    # Get first object ID
    if controller.store.objects:
        obj_id = list(controller.store.objects.keys())[0]
        try:
            revert_res = revert_object_refinement(obj_id)
            if hasattr(revert_res, 'size'):
                print(f"✅ revert_object_refinement returned image: {revert_res.size}") # type: ignore
            else:
                print(f"✅ revert_object_refinement returned: {revert_res}")
        except Exception as e:
            print(f"❌ revert_object_refinement failed: {e}")
            
    # 6. Test Export (UI Wrapper)
    print("\n🔹 Testing export_results...")
    try:
        export_msg = export_results("test_output")
        print(f"✅ export_results returned: {export_msg}")
    except Exception as e:
        print(f"❌ export_results failed: {e}")

if __name__ == "__main__":
    test_ui_flow()
