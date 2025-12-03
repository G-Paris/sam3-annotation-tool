
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

def test_sam3_video_loading():
    print("⏳ Loading Sam3VideoModel...")
    try:
        model = Sam3VideoModel.from_pretrained("facebook/sam3")
        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        print("✅ Sam3VideoModel loaded successfully (no warning expected about type mismatch).")
    except Exception as e:
        print(f"❌ Failed to load Sam3VideoModel: {e}")

if __name__ == "__main__":
    test_sam3_video_loading()
