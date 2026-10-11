"""English: Image-level splitting with optional fabric-roll grouping.
Hindi: Original images ka split, aur optional fabric-roll grouping.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import (
    DEFECT_DIR,
    MASK_DIR,
    NODEFECT_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    SPLIT_FILE,
    STRICT_GROUP_SPLIT,
    TEST_RATIO,
    VAL_RATIO,
)
from labels import class_of_image, fabric_code_of_image
from utils import find_mask, merged_mask


def collect_original_images() -> tuple:
    """English: Collect all original image paths.
    Hindi: Defect aur normal dono original images collect karta hai.
    """
    defect_images = sorted(DEFECT_DIR.rglob("*.png"))
    normal_images = sorted(NODEFECT_DIR.rglob("*.png"))
    return defect_images, normal_images


def audit_dataset(
    defect_images: List[Path],
    normal_images: List[Path],
) -> Dict:
    """English: Report dataset facts that affect how the split must be done.
    Hindi: Dataset ki important cheezein report karta hai, jaise missing masks,
    empty masks aur unparseable filenames.
    """
    without_mask: List[str] = []
    empty_mask: List[str] = []
    multi_mask: List[Dict] = []
    unparseable: List[str] = []

    for path in defect_images + normal_images:
        if class_of_image(path) is None:
            unparseable.append(path.name)

    for path in defect_images:
        paths = find_mask(path, MASK_DIR)
        if not paths:
            without_mask.append(path.name)
            continue
        if len(paths) > 1:
            multi_mask.append({"image": path.name, "masks": [p.name for p in paths]})
        if merged_mask(path, MASK_DIR) is None:
            empty_mask.append(path.name)

    counter: Counter = Counter()
    for path in defect_images:
        counter[class_of_image(path)] += 1
    counter["Normal"] = len(normal_images)

    rolls: Counter = Counter()
    for path in defect_images + normal_images:
        rolls[fabric_code_of_image(path)] += 1

    return {
        "defect_images": len(defect_images),
        "normal_images": len(normal_images),
        "total_classification_images": len(defect_images) + len(normal_images),
        "images_without_mask": without_mask,
        "images_with_empty_mask": empty_mask,
        "images_with_multiple_masks": multi_mask,
        "unparseable_filenames": unparseable,
        "class_distribution": dict(sorted(counter.items())),
        "fabric_roll_distribution": dict(sorted(rolls.items())),
        "distinct_fabric_rolls": len(rolls),
    }


def _allocate(count: int, val_ratio: float, test_ratio: float) -> tuple:
    """English: Decide how many val/test images a class of ``count`` gets.
    Hindi: Decide karta hai ki ek class ke kitne images val/test mein jayenge.

    English: ``sklearn.train_test_split(stratify=...)`` cannot be used here
    because ``Knots``, ``Contamination`` and ``Weft Crack`` contain a single
    original image each, which cannot satisfy a stratified split. The rules are:

    * count >= 3 -> at least one val and one test image
    * count == 2 -> one val image
    * count == 1 -> stays entirely in train
    """
    if count >= 3:
        n_val = max(1, int(round(count * val_ratio)))
        n_test = max(1, int(round(count * test_ratio)))
        while n_val + n_test >= count:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        return n_val, n_test
    if count == 2:
        return 1, 0
    return 0, 0


def stratified_group_split(
    paths: List[Path],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    """English: Split image paths per class while keeping rare classes usable.
    Hindi: Har class ke hisaab se split karta hai, aur rare classes ko
    train mein hi rakhta hai taaki unka koi matlab bache.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        by_class[class_of_image(path)].append(path)

    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    for class_name in sorted(by_class):
        members = list(by_class[class_name])
        rng.shuffle(members)
        n_val, n_test = _allocate(len(members), val_ratio, test_ratio)

        splits["val"].extend(members[:n_val])
        splits["test"].extend(members[n_val:n_val + n_test])
        splits["train"].extend(members[n_val + n_test:])

    for key in splits:
        splits[key].sort()

    return splits


def strict_roll_split(
    paths: List[Path],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    """English: Split so that no fabric roll spans more than one split.
    Hindi: Aisa split jo kisi fabric roll ko ek se zyada split mein nahi jaane
    deta.

    English: AITEX contains only 8 distinct fabric rolls, and defect types are
    strongly tied to specific rolls (``Knots`` exists only on roll ``05``,
    ``Weft Crack`` only on ``01``). Grouping by roll therefore removes most
    defect classes from validation and test. This mode is offered because it is
    the stricter protocol, not because it is usable for this dataset.
    Hindi: AITEX mein sirf 8 fabric rolls hain aur defect types particular
    rolls se jude hain. Isliye roll-wise grouping se zyadatar classes val/test
    se hat jaati hain. Ye mode isliye diya gaya hai kyunki ye protocol zyada
    strict hai, isliye nahi ki ye is dataset ke liye practical ho.
    """
    rng = np.random.default_rng(seed)
    by_roll: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        by_roll[fabric_code_of_image(path)].append(path)

    rolls = sorted(by_roll)
    rng.shuffle(rolls)

    total = len(paths)
    target_val = int(round(total * val_ratio))
    target_test = int(round(total * test_ratio))

    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    # English: Assign whole rolls greedily, always to the split that is furthest
    # below its target size. This keeps the roll intact while staying close to
    # the requested ratios.
    # Hindi: Poore rolls ko greedy assign karte hain, hamesha us split ko jo
    # apne target size se sabse door hai. Isse roll intact rehta hai aur ratio
    # bhi qareebi rehta hai.
    for roll in rolls:
        members = by_roll[roll]
        deficits = {
            "val": target_val - len(splits["val"]),
            "test": target_test - len(splits["test"]),
        }
        target = max(deficits, key=lambda key: deficits[key])
        if deficits[target] <= 0:
            target = "train"
        splits[target].extend(members)

    for key in splits:
        splits[key].sort()

    return splits


def summarise(splits: Dict[str, List[Path]]) -> Dict:
    """English: Build a per-split class and roll summary.
    Hindi: Har split ka class aur roll summary banata hai.
    """
    summary = {}
    for name, items in splits.items():
        classes = Counter(class_of_image(p) for p in items)
        rolls = Counter(fabric_code_of_image(p) for p in items)
        summary[name] = {
            "images": len(items),
            "class_distribution": dict(sorted(classes.items())),
            "fabric_rolls": sorted(rolls),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe image-level dataset split"
    )
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--strict-group-split",
        action="store_true",
        default=STRICT_GROUP_SPLIT,
        help=(
            "Keep every image from one fabric roll inside a single split. "
            "Hindi: Ek fabric roll ke saare images ek hi split mein."
        ),
    )
    args = parser.parse_args()

    defect_images, normal_images = collect_original_images()
    all_images = defect_images + normal_images

    print("English: Dataset audit")
    print("Hindi: Dataset ka audit")
    report = audit_dataset(defect_images, normal_images)
    print(json.dumps(report, indent=2))

    if report["unparseable_filenames"]:
        raise SystemExit(
            "Unparseable filenames found: " f"{report['unparseable_filenames']}"
        )

    if args.strict_group_split:
        splits = strict_roll_split(
            all_images, args.val_ratio, args.test_ratio, args.seed
        )
        mode = "strict_roll"
    else:
        splits = stratified_group_split(
            all_images, args.val_ratio, args.test_ratio, args.seed
        )
        mode = "class_stratified"

    print(f"\nSplit mode: {mode}")
    print("Split sizes:")
    for name, items in splits.items():
        print(f"  {name:<6} {len(items)}")

    print("\nClass distribution per split:")
    for name, items in splits.items():
        counter = Counter(class_of_image(p) for p in items)
        print(f"  {name}")
        for class_name, count in sorted(counter.items()):
            print(f"    {class_name:<16} {count}")
        print(f"    fabric rolls: {sorted({fabric_code_of_image(p) for p in items})}")

    if args.strict_group_split:
        roll_sets = {name: {fabric_code_of_image(p) for p in items}
                     for name, items in splits.items()}
        overlap = (roll_sets["train"] & roll_sets["val"]) | (
            roll_sets["train"] & roll_sets["test"]
        ) | (roll_sets["val"] & roll_sets["test"])
        print(f"\nRoll overlap between splits: {sorted(overlap) or 'none'}")
        if overlap:
            raise SystemExit("Roll grouping failed; aborting.")

    payload = {
        "seed": args.seed,
        "split_mode": mode,
        "strict_group_split": args.strict_group_split,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "audit": report,
        "summary": summarise(splits),
        "splits": {
            name: [str(path.relative_to(PROJECT_ROOT)) for path in items]
            for name, items in splits.items()
        },
        # English: Fabric roll per image, so a later grouped split or a
        # leakage audit does not have to re-derive it.
        # Hindi: Har image ka fabric roll, taaki baad mein grouped split ya
        # leakage audit ko ye dobara derive na karna pade.
        "image_rolls": {
            str(path.relative_to(PROJECT_ROOT)): fabric_code_of_image(path)
            for path in all_images
        },
    }

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"\nSaved split manifest to {SPLIT_FILE}")
    print(
        "Hindi: Original images split ho gayi. Ab inhi se patches banenge, "
        "isliye koi original image do alag sets mein nahi jaayegi."
    )


if __name__ == "__main__":
    main()
