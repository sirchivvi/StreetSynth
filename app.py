# app.py — HuggingFace Spaces entry point
import os
os.environ["LIGHTWEIGHT"] = "1"  # skip LaMa on CPU

from ui.app import demo

demo.launch()