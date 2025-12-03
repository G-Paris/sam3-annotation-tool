
import torch
import requests
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

def test_sam3_refinement():
    print("⏳ Loading Sam3TrackerModel...")
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print("✅ Model loaded.")

    # Create dummy image
    image = Image.new("RGB", (100, 100), "white")
    
    # Dummy inputs
    # 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
    input_points = [[[[50.0, 50.0]]]] 
    # 3 dimensions (image_dim, object_dim, point_label)
    input_labels = [[[1]]]
    
    print("🔹 Testing processor with input_points...")
    try:
        inputs = processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        )
        print("✅ Processor accepted input_points.")
    except Exception as e:
        print(f"❌ Processor failed: {e}")
        return

    print("🔹 Testing model inference...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Model inference successful.")
        
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        print(f"✅ Post-processing successful. Mask shape: {masks.shape}")
        
    except Exception as e:
        print(f"❌ Model inference failed: {e}")

if __name__ == "__main__":
    test_sam3_refinement()
