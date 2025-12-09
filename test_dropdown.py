import gradio as gr

def update_choices(text, history):
    if text and text not in history:
        history.append(text)
    return gr.update(choices=history, value=text), history

with gr.Blocks() as demo:
    history_state = gr.State(["cat", "dog"])
    
    dropdown = gr.Dropdown(
        choices=["cat", "dog"], 
        label="Test Dropdown", 
        allow_custom_value=True,
        interactive=True
    )
    
    btn = gr.Button("Submit")
    output = gr.Textbox(label="Output")
    
    btn.click(fn=update_choices, inputs=[dropdown, history_state], outputs=[dropdown, history_state])
    
    def print_val(val):
        return f"Selected: {val}"
        
    dropdown.change(fn=print_val, inputs=[dropdown], outputs=[output])

if __name__ == "__main__":
    demo.launch(show_error=True)
