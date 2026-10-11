"""English: Streamlit UI for fabric defect inspection.
Hindi: Fabric defect inspection ke liye Streamlit UI.

English: This file only handles the interface. Every inference decision lives in
``src/inference.py`` so that the CLI, the evaluation script and this app all
produce identical results.
Hindi: Ye file sirf UI handle karti hai. Saara inference logic
``src/inference.py`` mein hai, isliye CLI, evaluation aur app ka output same rehta hai.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    METRICS_DIR,
    PATCH_SIZE,
    STAGE1_DEFECT_PROB_THRESHOLD,
    STAGE1_MIN_DEFECTIVE_PATCHES,
)
from inference import (  # noqa: E402
    ModelNotTrainedError,
    load_classifier,
    load_segmenter,
    predict_image,
    stage1_scores,
    stage2_distribution,
)
from labels import DEFECT_TYPE_CLASSES, DEFECT_TYPE_INDEX_TO_CLASS  # noqa: E402
from ui_utils import (  # noqa: E402
    SAMPLES_DIR,
    generate_strip_svg,
    get_stage1_threshold_presets,
    load_json_file,
    parse_sample_filename,
)

st.set_page_config(
    page_title="Fabric Defect Inspection AI",
    page_icon="🧵",
    layout="wide",
)

MODEL_OPTIONS = {
    "EfficientNet-B0 (transfer learning)": "efficientnet",
    "Custom CNN (baseline)": "cnn",
}


@st.cache_resource(show_spinner="Loading models...")
def load_models(model_name: str):
    """English: Load the two classifiers and the segmenter once per session.
    Hindi: Dono classifiers aur segmenter ek hi baar load karta hai.
    """
    try:
        stage1, _ = load_classifier("binary", model_name, best=True)
        stage2, _ = load_classifier("defect_type", model_name, best=True)
    except (ModelNotTrainedError, ValueError, RuntimeError) as error:
        return None, None, None, str(error)

    try:
        segmenter, segmenter_checkpoint = load_segmenter(best=True)
    except (ModelNotTrainedError, ValueError, RuntimeError) as error:
        return stage1, stage2, None, str(error)

    return stage1, stage2, segmenter, segmenter_checkpoint


def get_top3_defect_types(
    stage2_model,
    image: np.ndarray,
    flagged_origins: List[Tuple[int, int]],
    flagged_weights: List[float],
) -> List[Tuple[str, float]]:
    """English: Compute top 3 defect types with probability percentages.
    Hindi: Top 3 defect types unki probabilities ke saath nikalta hai.
    """
    if stage2_model is None or not flagged_origins:
        return []

    from utils import pad_to_patch_grid, slice_patch  # src/utils
    padded = pad_to_patch_grid(image, PATCH_SIZE)
    patches = [slice_patch(padded, x, y, copy=True) for x, y in flagged_origins]

    dist, _ = stage2_distribution(stage2_model, patches, flagged_weights)
    if dist.size == 0:
        return []

    top_indices = np.argsort(dist)[::-1][:3]
    return [(DEFECT_TYPE_CLASSES[i], float(dist[i])) for i in top_indices]


def main() -> None:
    st.title("🧵 Fabric Defect Inspection AI")
    st.markdown(
        "**Two-stage Inspection & Localization System** | *AITEX Industrial Benchmark*\n\n"
        "English: Automatic fabric defect detection using deep learning: Stage 1 (binary defect filtering), "
        "Stage 2 (defect type categorization), and U-Net (pixel defect localization).\n\n"
        "Hindi: Deep learning dwara fabric inspection: Stage 1 (defect hai ya nahi), "
        "Stage 2 (defect ka type), aur U-Net (defect ki jagah)."
    )

    # Load dynamic test metrics for EfficientNet
    test_metrics = load_json_file(METRICS_DIR / "test_efficientnet.json")

    # Sidebar: Model and Decision Presets
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model_label = st.selectbox("Classification Model", list(MODEL_OPTIONS.keys()))
        model_name = MODEL_OPTIONS[selected_model_label]

        presets = get_stage1_threshold_presets(model_name)

        st.subheader("🎯 Decision Style")
        st.caption(
            "English: Choose trade-off between defect catch rate and false alarms.\n\n"
            "Hindi: Defect pakadne ki sensitivity chunein."
        )

        preset_choice = st.radio(
            "Presets",
            ["Balanced", "Sensitive", "Strict"],
            index=0,
            help=(
                "Balanced: validation-optimal balanced accuracy.\n"
                "Sensitive: prioritizes catching every defect (higher recall).\n"
                "Strict: minimizes false alarms (lower FPR)."
            ),
        )

        active_preset = presets[preset_choice]
        rec_str = f"{active_preset['recall'] * 100:.1f}%" if active_preset['recall'] is not None else "N/A"
        fpr_str = f"{active_preset['fpr'] * 100:.1f}%" if active_preset['fpr'] is not None else "N/A"
        bal_str = f"{active_preset['bal_acc'] * 100:.1f}%" if active_preset['bal_acc'] is not None else "N/A"

        st.info(
            f"**{preset_choice} Profile**\n"
            f"- Val Defect Recall: **{rec_str}**\n"
            f"- Val False Alarm Rate: **{fpr_str}**\n"
            f"- Val Balanced Accuracy: **{bal_str}**\n\n"
            f"*{active_preset['desc_hi']}*"
        )

        # Advanced expander for raw sliders
        with st.expander("🛠️ Advanced Threshold Tuning"):
            st.caption("Fine-tune raw thresholds manually:")
            tau_slider = st.slider(
                "P(defective) threshold (τ)",
                min_value=0.10,
                max_value=0.95,
                value=float(active_preset["tau"]),
                step=0.05,
                help="Hindi: Is probability se upar ke patch ko defective maana jayega.",
            )
            k_slider = st.number_input(
                "Min defective patches required (k)",
                min_value=1,
                max_value=16,
                value=int(active_preset["k"]),
                step=1,
                help="Hindi: Ek strip mein itne patches defective hone par poori image defective maani jayegi.",
            )
            unet_thresh_slider = st.slider(
                "U-Net mask threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Hindi: Pixel segmentation threshold.",
            )

        stage1_threshold_val = tau_slider
        min_patches_val = k_slider
        segmentation_threshold = unet_thresh_slider

        st.caption(
            f"**Active Thresholds:** τ = `{stage1_threshold_val:.2f}`, k = `{min_patches_val}`, U-Net τ = `{segmentation_threshold:.2f}`"
        )

        st.divider()
        st.subheader("🖼️ Sample Gallery")
        sample_files = [
            "None",
            "0010_006_02.png (Defect: Broken Yarn)",
            "0017_000_05.png (Normal fabric)",
            "0013_000_03.png (Normal - 9 of 16 patches exceeded threshold)",
            "0058_019_06.png (Defect: Fuzzyball)",
            "0020_016_03.png (Defect: Weft Curling)",
            "0018_000_06.png (Normal fabric)",
        ]
        sample_choice = st.selectbox("Try a sample image:", sample_files, index=0)

    # Model loading
    stage1_model, stage2_model, segmenter, load_error = load_models(model_name)
    if stage1_model is None or stage2_model is None:
        st.error(
            f"Trained checkpoints are missing or incompatible.\n\n{load_error}\n\n"
            "Please ensure checkpoints exist in models/classifier/."
        )
        st.stop()

    # Image source handling: Sample vs Upload
    image_bytes = None
    image_source_name = "uploaded_image.png"
    sample_meta = None

    if sample_choice != "None":
        filename = sample_choice.split(" ")[0]
        sample_path = SAMPLES_DIR / filename
        if sample_path.exists():
            image_bytes = sample_path.read_bytes()
            image_source_name = filename
            sample_meta = parse_sample_filename(filename)

    uploaded = st.file_uploader(
        "Upload fabric strip image (or use sample above)",
        type=["png", "jpg", "jpeg"],
        help="Recommended size: 256 x 4096 pixels.",
    )

    if uploaded is not None:
        if uploaded.size == 0:
            st.error(
                "The uploaded file is empty (0 bytes). Please upload a valid fabric image.\n\n"
                "Hindi: Upload ki gayi file khaali hai (0 bytes). Kripya valid image upload karein."
            )
            st.stop()
        image_bytes = uploaded.read()
        image_source_name = uploaded.name
        sample_meta = parse_sample_filename(uploaded.name)

    if image_bytes is None:
        st.info(
            "👆 Upload a fabric image or select one from the **Sample Gallery** in the sidebar to begin.\n\n"
            "Hindi: Shuru karne ke liye image upload karein ya sidebar se sample chunein."
        )
        st.stop()

    # Validate image decoding
    try:
        import io
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(image_bytes))
        array = np.array(pil_img.convert("RGB"))
    except Exception as error:
        st.error(
            f"Could not read or process the uploaded file as an image ({error}). "
            "Please ensure you uploaded a valid, uncorrupted PNG or JPEG image.\n\n"
            f"Hindi: File ko image ke roop mein nahi padha ja saka ({error}). "
            "Kripya sahi PNG ya JPEG image upload karein."
        )
        st.stop()

    if array.size == 0 or array.ndim != 3 or array.shape[2] != 3:
        st.error(
            f"Expected a non-empty RGB image, received shape {array.shape}.\n\n"
            "Hindi: Sahi RGB image format chahiye."
        )
        st.stop()

    height, width = array.shape[:2]

    # Non-blocking dimension warning
    if height != 256 or not (3700 <= width <= 4150):
        st.warning(
            f"⚠️ **Image Dimensions Note**: Input image is {width}×{height} px. "
            f"Standard AITEX strips are 256×4096 px. The model will automatically pad/slice "
            f"patches, but non-standard textures may affect detection accuracy.\n\n"
            f"Hindi: Image size standard (256×4096) se alag hai, model isse pad karega."
        )

    # Run inference pipeline
    with st.spinner("Analyzing fabric strip across all patches..."):
        try:
            scores_data = stage1_scores(stage1_model, array)
            result = predict_image(
                array,
                stage1_model,
                stage2_model,
                segmenter=segmenter,
                stage1_threshold=stage1_threshold_val,
                min_defective_patches=min_patches_val,
                segmentation_threshold=segmentation_threshold,
            )
        except Exception as error:
            st.error(f"Inference failed: {error}")
            st.stop()

    # Extract patch probabilities and flagged items
    patch_scores_list = [(x, y, s) for x, y, s in scores_data]
    flagged_patches = [(x, y, s) for x, y, s in scores_data if s >= stage1_threshold_val]
    flagged_count = len(flagged_patches)
    total_patches = len(scores_data)

    # Top-3 defect types if defective
    top3_types = []
    if result.is_defective and stage2_model is not None:
        flagged_origins = [(x, y) for x, y, _ in flagged_patches]
        flagged_weights = [s for _, _, s in flagged_patches]
        top3_types = get_top3_defect_types(stage2_model, array, flagged_origins, flagged_weights)

    # Create UI Tabs
    tab_result, tab_loc, tab_patches, tab_info = st.tabs([
        "📊 Result",
        "🎯 Localization",
        "🔍 Patch Details",
        "ℹ️ How It Works",
    ])

    # ---------------- TAB 1: RESULT ----------------
    with tab_result:
        # Ground-truth banner for demo samples
        if sample_meta and sample_meta["expected_class"] != "Unknown":
            expected_def = sample_meta["is_defective"]
            model_def = result.is_defective
            match = (expected_def == model_def)
            icon = "✅" if match else "❌"
            status_text = "Match" if match else "Discrepancy"

            st.markdown(
                f"<div style='background:#1e2430; padding:10px 16px; border-radius:8px; margin-bottom:12px; border-left:4px solid {'#00e676' if match else '#ff5252'}'>"
                f"<b>Sample Ground Truth ({image_source_name}):</b> Expected: <code>{sample_meta['expected_class']}</code> | "
                f"Model verdict: <code>{'Defective' if model_def else 'Normal'}</code> "
                f"({icon} {status_text})<br>"
                f"<span style='color:#a0a0a0; font-size:12px;'>*Demonstration sample from AITEX benchmark.*</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Big Verdict Banner
        if result.is_defective:
            st.error("### 🚨 DEFECT DETECTED\n**Defective Fabric Strip** | Defect presence confirmed by Stage 1 classifier.")
        else:
            st.success("### ✅ NO DEFECT FOUND\n**Normal Fabric Strip** | Strip passes Stage 1 defect criteria.")

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Stage 1 Defect Probability",
                value=f"{result.stage1_defect_probability * 100:.1f}%",
                help="Aggregated P(defective) across flagged patches",
            )
        with col2:
            st.metric(
                label="Flagged Patches",
                value=f"{flagged_count} of {total_patches}",
                delta=f"Need ≥ {min_patches_val} for defect",
                delta_color="normal" if result.is_defective else "off",
            )
        with col3:
            pixel_ratio = float(result.mask.mean() * 100) if (result.mask is not None and result.is_defective) else 0.0
            st.metric(
                label="Defect Pixel Ratio",
                value=f"{pixel_ratio:.3f}%" if result.is_defective else "0.000%",
                help="Estimated fraction of strip area occupied by defect pixels",
            )
        with col4:
            st.metric(
                label="Active Thresholds",
                value=f"τ = {stage1_threshold_val:.2f}",
                delta=f"min k = {min_patches_val}",
                delta_color="off",
            )

        st.markdown("---")

        # Strip Diagram
        st.subheader("Full Fabric Strip Patch Diagram")
        st.caption(
            "English: Horizontal diagram of all 16 consecutive 256×256 patches. "
            "Colors indicate defect probability (blue = clean, red = high defect likelihood). "
            "Flagged patches are outlined with thick red borders (★).\n\n"
            "Hindi: 16 patches ka chart. Laal rang aur star (★) defect wale patches ko darshata hai."
        )
        svg_diagram = generate_strip_svg(patch_scores_list, stage1_threshold_val, min_patches_val)
        st.components.v1.html(svg_diagram, height=95)

        # Defect Type Suggestion (Only for Defective images)
        if result.is_defective:
            st.markdown("---")
            st.subheader("💡 Suggested Defect Type")
            st.caption(
                "English: Defect type is an automated suggestion, not an absolute verdict. "
                "Because severe class imbalance exists in AITEX, Stage 2 often defaults towards Broken Pick.\n\n"
                "Hindi: Defect type ek sujhav (suggestion) hai, pakka verdict nahi."
            )

            if top3_types:
                chart_data = pd.DataFrame(
                    [{"Defect Type": name, "Probability (%)": prob * 100} for name, prob in top3_types]
                ).sort_values("Probability (%)", ascending=True)

                st.bar_chart(chart_data.set_index("Defect Type"), horizontal=True)

                top_prob = top3_types[0][1]
                if top_prob < 0.40:
                    st.warning(
                        "⚠️ **Low Confidence: Defect type is an uncertain guess (< 40%).**\n\n"
                        "English: The classifier lacks high certainty on this defect category. Verify visually.\n\n"
                        "Hindi: Model is defect category ko lekar confident nahi hai (< 40%). Kripya khud dekhein."
                    )

            # Note from test_efficientnet.json
            if test_metrics:
                dt_test = test_metrics.get("stages", {}).get("image_level", {}).get("defect_type", {})
                bal_acc_val = dt_test.get("balanced_accuracy", 0.325)
                st.caption(
                    f"ℹ️ *Empirical note from test benchmark:* Defect-type test balanced accuracy is **{bal_acc_val * 100:.1f}%**. "
                    "Classes like Broken Pick dominate the distribution while rare classes (e.g. Broken Yarn, Cut Selvage) have limited support."
                )
        else:
            st.info(
                "ℹ️ **Clean Strip:** Since no defect was detected, Stage 2 defect type categorization "
                "and segmentation are omitted.\n\n"
                "Hindi: Image Normal hai, isliye defect type aur mask skip kiye gaye hain."
            )

    # ---------------- TAB 2: LOCALIZATION ----------------
    with tab_loc:
        if not result.is_defective:
            st.info(
                "No defects detected. Localization is only available for defective fabrics.\n\n"
                "Hindi: Defect milne par hi localization mask dikhaya jata hai."
            )
        elif result.mask is None:
            st.warning("U-Net segmentation checkpoint is not loaded or segmentation was skipped.")
        else:
            st.subheader("Defect Localization (U-Net)")
            st.caption(
                "English: Preliminary: the U-Net localizes some defects well and misses others. "
                "Bounding boxes are derived directly from the segmentation mask, not an object detector.\n\n"
                "Hindi: U-Net defect ki jagah highlight karta hai."
            )

            # Benchmark stats from test_efficientnet.json
            if test_metrics:
                seg_test = test_metrics.get("stages", {}).get("segmentation_image_level", {})
                touch_rate = seg_test.get("box_touch_rate", 0.5714)
                mean_dice = seg_test.get("mean_dice", 0.1575)
                st.info(
                    f"📊 **Benchmark Test Localization Quality:** Box-touch rate = **{touch_rate * 100:.1f}%** (8/14 images), "
                    f"Mean Dice = **{mean_dice:.4f}**. Fine yarn defects are challenging for pixel segmentation."
                )

            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                show_overlay = st.checkbox("Show Semi-Transparent Mask Overlay", value=True)
            with col_ctrl2:
                show_bbox = st.checkbox("Draw Bounding Box on Overlay", value=True)

            # Display visual
            vis_image = array.copy()
            if show_overlay and result.overlay is not None:
                vis_image = result.overlay.copy()

            if show_bbox and result.bounding_box:
                bb = result.bounding_box
                cv2.rectangle(
                    vis_image,
                    (bb["x_min"], bb["y_min"]),
                    (bb["x_max"], bb["y_max"]),
                    (0, 255, 255),
                    2,
                )

            st.image(vis_image, caption="Defect Localization View", use_container_width=True)

            if result.bounding_box:
                bb = result.bounding_box
                st.write(
                    f"**Detected Defect Region:** x = `[{bb['x_min']} .. {bb['x_max']}]`, "
                    f"y = `[{bb['y_min']} .. {bb['y_max']}]` "
                    f"(width: `{bb['width']}px`, height: `{bb['height']}px`)"
                )

            with st.expander("Show Raw Binary Mask"):
                st.image(
                    (result.mask * 255).astype(np.uint8),
                    caption="Raw U-Net Binary Mask (White = Defect)",
                    use_container_width=True,
                )

    # ---------------- TAB 3: PATCH DETAILS ----------------
    with tab_patches:
        st.subheader("Patch-by-Patch Inspection Table")
        st.caption(
            "English: Breakdown of all 256×256 patches extracted across the fabric strip.\n\n"
            "Hindi: Har patch ka detail table."
        )

        rows = []
        for i, (x, y, prob) in enumerate(scores_data):
            is_fl = prob >= stage1_threshold_val
            # Find matching patch_prediction if any
            p_pred = next((p for p in result.patch_predictions if p.x == x and p.y == y), None)
            pix_ratio = p_pred.defect_pixel_ratio if p_pred else None

            rows.append({
                "Patch #": i + 1,
                "Origin X": x,
                "Origin Y": y,
                "Defect Prob P(defect)": round(prob, 4),
                "Status": "🚨 FLAGGED" if is_fl else "Normal",
                "Defect Pixel Ratio": f"{pix_ratio * 100:.3f}%" if pix_ratio is not None else "-",
            })

        df_patches = pd.DataFrame(rows)

        # Style table with color gradient on probability
        def highlight_prob(val):
            color = "#2a1515" if val >= stage1_threshold_val else "transparent"
            font_weight = "bold" if val >= stage1_threshold_val else "normal"
            return f"background-color: {color}; font-weight: {font_weight}"

        styled_df = df_patches.style.map(highlight_prob, subset=["Defect Prob P(defect)"])
        st.dataframe(styled_df, use_container_width=True)

    # ---------------- TAB 4: HOW IT WORKS ----------------
    with tab_info:
        st.subheader("Architecture & Method Overview")
        st.markdown(
            """
### 🔬 How the Pipeline Operates
1. **Patch Extraction**: A long fabric strip (e.g. 256×4096 px) is split into a sequence of sixteen consecutive 256×256 patches.
2. **Stage 1 (Binary Screening)**: An EfficientNet-B0 classifier screens each patch independently for defect presence.
   - If at least $k$ patches exceed probability threshold $\\tau$, the image is declared **Defective**.
3. **Stage 2 (Defect Type Suggestion)**: Flagged defective patches are processed by a 12-class classifier, aggregating softmax predictions into a suggested defect category.
4. **Localization (U-Net)**: A convolutional U-Net segments the flagged patches at pixel resolution, producing an overlay mask and bounding box.

---
### ⚙️ What the Decision Thresholds Mean
- **$\\tau$ (P(defective) Threshold)**: The probability boundary a patch must reach to be counted as defective.
- **$k$ (Minimum Defective Patches)**: Number of patches that must be flagged to call the entire fabric defective.
- Higher $\\tau$ and $k$ $\\implies$ **Strict Mode** (fewer false alarms, but might miss small subtle defects).
- Lower $\\tau$ and $k$ $\\implies$ **Sensitive Mode** (catches nearly all defects, higher false positive rate).
            """
        )

        st.subheader("⚠️ Benchmark Limitations & Honest Assessment")
        if test_metrics:
            il_bin = test_metrics.get("stages", {}).get("image_level", {}).get("binary", {})
            rec_val = il_bin.get("defect_recall", 0.875)
            rec_ci = il_bin.get("defect_recall_ci95", [0.6398, 0.9650])
            fpr_val = il_bin.get("normal_false_positive_rate", 0.0476)
            fpr_ci = il_bin.get("normal_fpr_ci95", [0.0085, 0.2267])
            bal_val = il_bin.get("balanced_accuracy", 0.9137)
            bal_ci = il_bin.get("balanced_accuracy_ci95", [0.7066, 0.9783])

            st.markdown(
                f"""
<div style='background:#1b222d; padding:16px; border-radius:8px; border:1px solid #2e3846;'>
<b>Empirical Test Metrics (37 Independent Test Images):</b>
<ul>
  <li><b>Defect Recall:</b> {rec_val * 100:.1f}% (Wilson 95% CI: [{rec_ci[0] * 100:.1f}%, {rec_ci[1] * 100:.1f}%])</li>
  <li><b>Normal False Alarm Rate (FPR):</b> {fpr_val * 100:.1f}% (Wilson 95% CI: [{fpr_ci[0] * 100:.1f}%, {fpr_ci[1] * 100:.1f}%])</li>
  <li><b>Balanced Accuracy:</b> {bal_val * 100:.1f}% (Wilson 95% CI: [{bal_ci[0] * 100:.1f}%, {bal_ci[1] * 100:.1f}%])</li>
</ul>
<b>Known Limitations:</b>
<ol>
  <li><b>Small Benchmark Dataset:</b> The entire AITEX dataset contains only 247 original images (141 normal, 106 defective). Due to the small test sample size (37 images), statistical confidence intervals are wide.</li>
  <li><b>Defect Type Imbalance:</b> Rare classes (e.g. Contamination, Knots, Weft Crack) have single-image support; Stage 2 classification is an advisory guess.</li>
  <li><b>Preliminary Localization:</b> The U-Net was trained without pretraining on a small patch sample; pixel-level Dice (~0.158) is preliminary.</li>
</ol>
</div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- DOWNLOAD SECTION ----------------
    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        # Download overlay image
        if result.overlay is not None:
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(result.overlay, cv2.COLOR_RGB2BGR))
            if is_success:
                st.download_button(
                    label="📥 Download Overlay Image (PNG)",
                    data=buffer.tobytes(),
                    file_name=f"overlay_{image_source_name}",
                    mime="image/png",
                )
        else:
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            if is_success:
                st.download_button(
                    label="📥 Download Inspected Image (PNG)",
                    data=buffer.tobytes(),
                    file_name=f"inspected_{image_source_name}",
                    mime="image/png",
                )

    with col_dl2:
        # Download machine-readable JSON
        summary_payload = result.summary()
        summary_payload["thresholds_used"] = {
            "stage1_probability_threshold": stage1_threshold_val,
            "min_defective_patches": min_patches_val,
            "segmentation_threshold": segmentation_threshold,
        }
        summary_payload["patch_probabilities"] = [
            {"patch_idx": idx + 1, "x": x, "y": y, "probability": prob, "flagged": prob >= stage1_threshold_val}
            for idx, (x, y, prob) in enumerate(scores_data)
        ]
        if top3_types:
            summary_payload["top3_defect_type_suggestions"] = [
                {"defect_class": name, "probability": prob} for name, prob in top3_types
            ]

        st.download_button(
            label="📥 Download Inspection Report (JSON)",
            data=json.dumps(summary_payload, indent=2),
            file_name=f"report_{Path(image_source_name).stem}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
