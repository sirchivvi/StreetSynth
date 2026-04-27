# app.py — HuggingFace Spaces entry point
import os
os.environ["LIGHTWEIGHT"] = "1"

from ui.app import demo

demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)