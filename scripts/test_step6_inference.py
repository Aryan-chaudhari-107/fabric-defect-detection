import sys
import json
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DEVICE
from inference import load_classifier, load_segmenter, predict_image, stage1_scores

def run():
    s1_model, _ = load_classifier('binary', 'efficientnet', best=True, device=DEVICE)
    s2_model, _ = load_classifier('defect_type', 'efficientnet', best=True, device=DEVICE)
    segmenter, _ = load_segmenter(device=DEVICE)

    with open('results/metrics/stage1_thresholds_efficientnet.json') as f:
        thresh_data = json.load(f)
    tau = thresh_data['best_stage1_prob_threshold']
    k = thresh_data['best_min_defective_patches']

    img_paths = [
        'data/raw/Defect_images/0010_006_02.png',
        'data/raw/NODefect_images/2306881-210020u/0017_000_05.png'
    ]

    for p in img_paths:
        img = cv2.imread(p)
        scores = stage1_scores(s1_model, img, device=DEVICE)
        res = predict_image(
            img,
            s1_model,
            s2_model,
            segmenter=segmenter,
            stage1_threshold=tau,
            min_defective_patches=k
        )
        probs = [s for _, _, s in scores]
        flagged = [s for s in probs if s >= tau]
        verdict = "Defective" if res.is_defective else "Normal"
        print(f"=== Image: {p} ===")
        print("Stage 1 Patch Probabilities (16 patches):")
        for i, (x, y, s) in enumerate(scores):
            print(f"  Patch {i:02d} (x={x:04d}, y={y:04d}): P(defect)={s:.4f}")
        print("Summary:")
        print(f"  Stage 1 Defect Probability (aggregated): {res.stage1_defect_probability:.4f}")
        print(f"  Max Patch Probability: {max(probs):.4f}")
        print(f"  Flagged Patches (>= {tau}): {len(flagged)}")
        print(f"  Min Patches Required (k): {k}")
        print(f"  Verdict: {verdict}")
        print(f"  Defect Class: {res.defect_class}")
        print(f"  Confidence: {res.confidence:.4f}")
        print(f"  Bounding Box: {res.bounding_box}")
        print()

if __name__ == '__main__':
    run()
