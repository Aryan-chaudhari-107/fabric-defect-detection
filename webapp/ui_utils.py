"""English: Reusable helper functions for the Streamlit webapp.
Hindi: Streamlit webapp ke liye reusable helper functions.

English: Reads metrics dynamically from results/metrics/*.json, renders
SVG strip patch diagrams, generates downloadable files and parses filenames.
Hindi: JSON files se dynamic metrics load karta hai, SVG diagrams banata hai,
downloads handle karta hai.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Path constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SAMPLES_DIR = PROJECT_ROOT / "webapp" / "assets" / "samples"

AITEX_DEFECT_CODES = {
    "000": "Normal",
    "002": "Broken End",
    "006": "Broken Yarn",
    "010": "Broken Pick",
    "016": "Weft Curling",
    "019": "Fuzzyball",
    "022": "Cut Selvage",
    "023": "Crease",
    "025": "Warp Ball",
    "027": "Knots",
    "029": "Contamination",
    "030": "Nep",
    "036": "Weft Crack",
}


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """English: Safely load a JSON file, returning None on missing or corrupt files.
    Hindi: JSON file ko safely load karta hai.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_stage1_threshold_presets(model_name: str) -> Dict[str, Dict[str, Any]]:
    """English: Compute Balanced, Sensitive and Strict decision presets from stage1_thresholds_<model>.json.
    Hindi: stage1_thresholds file se Balanced, Sensitive aur Strict presets nikalta hai.
    """
    file_path = METRICS_DIR / f"stage1_thresholds_{model_name}.json"
    data = load_json_file(file_path)

    presets = {
        "Balanced": {
            "tau": 0.60,
            "k": 2,
            "recall": None,
            "fpr": None,
            "bal_acc": None,
            "description": "Validation-optimal balanced accuracy",
            "desc_hi": "Validation par sabse behtareen balanced accuracy",
        },
        "Sensitive": {
            "tau": 0.45,
            "k": 3,
            "recall": None,
            "fpr": None,
            "bal_acc": None,
            "description": "Catches almost all defects (higher false alarms)",
            "desc_hi": "Zyada se zyada defect pakadne ke liye (kuch false alarm ho sakte hain)",
        },
        "Strict": {
            "tau": 0.70,
            "k": 2,
            "recall": None,
            "fpr": None,
            "bal_acc": None,
            "description": "Fewer false alarms (might miss subtle defects)",
            "desc_hi": "Kam false alarm (par chhota defect miss ho sakta hai)",
        },
    }

    if not data:
        return presets

    # 1. Balanced: best_stage1_prob_threshold and best_min_defective_patches
    presets["Balanced"]["tau"] = float(data.get("best_stage1_prob_threshold", 0.60))
    presets["Balanced"]["k"] = int(data.get("best_min_defective_patches", 2))
    presets["Balanced"]["bal_acc"] = data.get("val_balanced_accuracy")
    presets["Balanced"]["recall"] = data.get("val_defect_recall")
    presets["Balanced"]["fpr"] = data.get("val_normal_false_positive_rate")

    candidates = data.get("top_candidates", [])
    if candidates:
        # 2. Sensitive: highest defect_recall, tie: lowest normal_fpr
        sens = sorted(candidates, key=lambda c: (c["defect_recall"], -c["normal_fpr"]), reverse=True)[0]
        presets["Sensitive"]["tau"] = float(sens["stage1_prob_threshold"])
        presets["Sensitive"]["k"] = int(sens["min_defective_patches"])
        presets["Sensitive"]["bal_acc"] = sens.get("balanced_accuracy")
        presets["Sensitive"]["recall"] = sens.get("defect_recall")
        presets["Sensitive"]["fpr"] = sens.get("normal_fpr")

        # 3. Strict: lowest normal_fpr, tie: highest defect_recall
        strict = sorted(candidates, key=lambda c: (-c["normal_fpr"], c["defect_recall"]), reverse=True)[0]
        presets["Strict"]["tau"] = float(strict["stage1_prob_threshold"])
        presets["Strict"]["k"] = int(strict["min_defective_patches"])
        presets["Strict"]["bal_acc"] = strict.get("balanced_accuracy")
        presets["Strict"]["recall"] = strict.get("defect_recall")
        presets["Strict"]["fpr"] = strict.get("normal_fpr")

    return presets


def parse_sample_filename(filename: str) -> Dict[str, str]:
    """English: Parse expected defect info from AITEX sample filename <id>_<defect_code>_<roll>.
    Hindi: AITEX filename se expected defect information nikalta hai.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        code = parts[1]
        defect_class = AITEX_DEFECT_CODES.get(code, "Unknown")
        is_defective = (code != "000")
        return {
            "image_id": parts[0],
            "defect_code": code,
            "roll": parts[2],
            "expected_class": defect_class,
            "is_defective": is_defective,
        }
    return {
        "image_id": stem,
        "defect_code": "unknown",
        "roll": "unknown",
        "expected_class": "Unknown",
        "is_defective": False,
    }


def generate_strip_svg(
    patch_scores: List[Tuple[int, int, float]],
    tau: float,
    k: int,
    width: int = 1000,
    height: int = 70,
) -> str:
    """English: Generate an inline HTML/SVG diagram displaying all 256x256 patches in order.
    Coloured by Stage 1 defect probability with red outline on flagged patches.
    Hindi: Saare 16 patches ka horizontal SVG strip diagram banata hai.
    """
    num_patches = max(len(patch_scores), 1)
    patch_width = width / num_patches

    def prob_to_color(p: float) -> str:
        # Interpolate from cool blue/gray (0.0) -> orange (0.5) -> bright red (1.0)
        p = max(0.0, min(1.0, p))
        if p < 0.5:
            # Blue-green to orange
            factor = p / 0.5
            r = int(50 + factor * 190)
            g = int(120 + factor * 40)
            b = int(210 - factor * 170)
        else:
            # Orange to deep red
            factor = (p - 0.5) / 0.5
            r = int(240 + factor * 15)
            g = int(160 - factor * 120)
            b = int(40 - factor * 20)
        return f"rgb({r},{g},{b})"

    rects = []
    labels = []

    for i, (x, y, prob) in enumerate(patch_scores):
        x_pos = i * patch_width
        fill_col = prob_to_color(prob)
        is_flagged = prob >= tau
        stroke = "#ff1744" if is_flagged else "#333333"
        stroke_width = "3.5" if is_flagged else "1"
        flagged_tag = " [FLAGGED]" if is_flagged else ""

        tooltip = f"Patch {i + 1} of {num_patches}&#10;Origin x={x}, y={y}&#10;P(defect)={prob:.4f}{flagged_tag}"

        rects.append(
            f'<rect x="{x_pos:.1f}" y="2" width="{patch_width - 2:.1f}" height="{height - 24}" '
            f'fill="{fill_col}" stroke="{stroke}" stroke-width="{stroke_width}" rx="3">'
            f'<title>{tooltip}</title></rect>'
        )

        text_x = x_pos + (patch_width / 2.0)
        prob_percent = f"{prob * 100:.0f}%"
        flag_star = "★ " if is_flagged else ""
        labels.append(
            f'<text x="{text_x:.1f}" y="{height - 10}" text-anchor="middle" font-size="11" '
            f'fill="#e0e0e0" font-family="sans-serif" font-weight="{"bold" if is_flagged else "normal"}">'
            f'{flag_star}P{i + 1}</text>'
        )
        labels.append(
            f'<text x="{text_x:.1f}" y="{height / 2 - 2:.1f}" text-anchor="middle" font-size="11" '
            f'fill="#ffffff" font-family="sans-serif" font-weight="bold" pointer-events="none">'
            f'{prob_percent}</text>'
        )

    svg_content = (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#181b20; border-radius:6px; padding:4px;">'
        + "".join(rects)
        + "".join(labels)
        + "</svg>"
    )
    return svg_content
