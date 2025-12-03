import gradio as gr
from PIL import Image

def on_select(evt: gr.SelectData):
    print(f"Selected: {evt.index} value: {evt.value}")
    return f"Clicked at {evt.index}"

with gr.Blocks() as demo:
    # Create a dummy image
    img = Image.new("RGB", (500, 500), "white")
    
    # Test ImageEditor
    im_ed = gr.ImageEditor(value=img, type="pil", height=400)
    out = gr.Textbox()
    
    im_ed.select(on_select, outputs=out)

if __name__ == "__main__":
    demo.launch()
