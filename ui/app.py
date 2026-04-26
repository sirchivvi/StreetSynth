import gradio as gr
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import StreetSynthPipeline

# ── Load pipeline once at startup ─────────────────────────────────────────
CHECKPOINT = os.environ.get("CHECKPOINT_PATH", None)
LIGHTWEIGHT = os.environ.get("LIGHTWEIGHT", "0") == "1"

print("Initializing StreetSynth...")
pipeline = StreetSynthPipeline(
    checkpoint_path=CHECKPOINT,
    lightweight=LIGHTWEIGHT
)

# ── Inference function ────────────────────────────────────────────────────
def run_streetsynth(image, intervention_type, show_depth, show_segmentation):
    if image is None:
        return None, None, None, "Please upload an image."

    # Save temp input
    from PIL import Image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    Image.fromarray(image).save(tmp_path)

    result = pipeline.run(
        image_path=tmp_path,
        intervention_type=intervention_type,
        output_dir="/tmp/streetsynth_outputs"
    )
    os.unlink(tmp_path)

    if not result["valid"]:
        return None, None, None, f"❌ {result['reason']}"

    # Format indicators
    ind = result["indicators"]
    score = ind["overall"]["score"]
    stars = "⭐" * score + "☆" * (3 - score)
    indicator_text = f"""
## Accessibility Score: {stars} ({score}/3)

| Indicator | Status | Detail |
|-----------|--------|--------|
| 🚶 Connectivity | {"✅" if ind["connectivity"]["score"] else "❌"} | {ind["connectivity"]["reason"]} |
| 🪑 Comfort      | {"✅" if ind["comfort"]["score"]      else "❌"} | {ind["comfort"]["reason"]}      |
| ♿ Mobility     | {"✅" if ind["mobility"]["score"]     else "❌"} | {ind["mobility"]["reason"]}     |

**Placement:** Center=({result["placement"]["x"]}, {result["placement"]["y"]}) — {result["placement"]["reason"]}
    """

    before = result["img_np"]
    after  = result["final_img"]

    # Optional overlays
    seg_overlay = None
    if show_segmentation:
        from cv.segmentation import PALETTE
        seg_color   = np.array(PALETTE, dtype=np.uint8)[result["seg_map"]]
        seg_overlay = (before * 0.5 + seg_color * 0.5).astype(np.uint8)

    depth_overlay = None
    if show_depth:
        import matplotlib.cm as cm
        depth_colored = (cm.plasma(result["depth_norm"])[:,:,:3] * 255).astype(np.uint8)
        depth_overlay = (before * 0.5 + depth_colored * 0.5).astype(np.uint8)

    extra = seg_overlay if show_segmentation else (depth_overlay if show_depth else None)

    return before, after, extra, indicator_text


# ── UI Layout ─────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #666; }
.gradio-container { max-width: 1200px; margin: auto; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🏙️ StreetSynth", elem_id="title")
    gr.Markdown(
        "**Human-in-the-Loop GAN for Inclusive Street Design** — "
        "Upload a street photo and visualize accessibility interventions.",
        elem_id="subtitle"
    )

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Input")
            image_input = gr.Image(
                label="Street Photo",
                type="numpy",
                height=300
            )

            gr.Markdown("### ⚙️ Settings")
            intervention = gr.Radio(
                choices=["crosswalk", "bench", "curb_ramp"],
                value="crosswalk",
                label="Intervention Type",
                info="Select the accessibility feature to add"
            )

            with gr.Accordion("Advanced Options", open=False):
                show_seg   = gr.Checkbox(label="Show segmentation overlay", value=False)
                show_depth = gr.Checkbox(label="Show depth overlay",        value=False)

            run_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

        # ── Right column: outputs ─────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Results")
            with gr.Row():
                before_out = gr.Image(label="Before", height=250)
                after_out  = gr.Image(label="After",  height=250)

            extra_out    = gr.Image(label="Overlay (seg/depth)", height=250, visible=True)
            indicator_md = gr.Markdown("*Results will appear here after generation.*")

    # ── Examples ──────────────────────────────────────────────────────────
    gr.Markdown("### 🗺️ Try an Example")
    gr.Examples(
        examples=[
            ["test_images/sample.jpg", "crosswalk", False, False],
        ],
        inputs=[image_input, intervention, show_seg, show_depth],
        label="Example inputs"
    )

    # ── Footer ────────────────────────────────────────────────────────────
    gr.Markdown(
        "---\n"
        "**StreetSynth** | Akhil Chivukula | AIML A1 | "
        "[GitHub](https://github.com/sirchivvi/StreetSynth)"
    )

    # ── Wire up ───────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_streetsynth,
        inputs=[image_input, intervention, show_depth, show_seg],
        outputs=[before_out, after_out, extra_out, indicator_md]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False   # set True for temporary public URL
    )