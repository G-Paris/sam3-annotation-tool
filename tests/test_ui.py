import gradio as gr

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# UI Navigation Test (Tabs)")
        
        # State
        current_index = gr.State(0)
        total_images = gr.State(3)

        # We use a Tabs component to manage screens. 
        # We can hide the tabs header with CSS if we want a pure wizard feel later.
        with gr.Tabs() as tabs:
            with gr.TabItem("Setup", id=0) as tab_setup:
                gr.Markdown("## Setup Screen")
                start_btn = gr.Button("Start")

            with gr.TabItem("Input", id=1) as tab_input:
                gr.Markdown("## Input Screen")
                img_display = gr.Textbox(label="Current Image", value="Image 0")
                run_btn = gr.Button("Run Inference")

            with gr.TabItem("Result", id=2) as tab_result:
                gr.Markdown("## Result Screen")
                confirm_btn = gr.Button("Confirm & Edit")

            with gr.TabItem("Editor", id=3) as tab_editor:
                gr.Markdown("## Editor Screen")
                finish_btn = gr.Button("Finish & Next")

        # --- LOGIC ---

        def start():
            return gr.update(selected=1)

        start_btn.click(fn=start, outputs=[tabs])

        def run_inference():
            return gr.update(selected=2)

        run_btn.click(fn=run_inference, outputs=[tabs])

        def confirm():
            return gr.update(selected=3)

        confirm_btn.click(fn=confirm, outputs=[tabs])

        def finish_and_next(idx, total):
            next_idx = idx + 1
            if next_idx < total:
                print(f"Moving to image {next_idx}")
                # Update data AND switch tab
                return next_idx, f"Image {next_idx}", gr.update(selected=1)
            else:
                print("Finished")
                return idx, "Finished", gr.update(selected=3)

        finish_btn.click(
            fn=finish_and_next,
            inputs=[current_index, total_images],
            outputs=[current_index, img_display, tabs]
        )

    return demo

def test_ui():
    demo = create_demo()
    assert demo is not None

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(show_error=True)
