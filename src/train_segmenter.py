"""English: Train the U-Net segmentation model.
Hindi: Defect mask predict karne ke liye U-Net train hota hai.
"""

import argparse

from config import (
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    SEGMENTER_BATCH_SIZE,
    SEGMENTER_EPOCHS,
    SEGMENTER_LEARNING_RATE,
)
from dataset import SegmentationPatchDataset, load_split
from segmentation_training import train_segmenter
from training import set_seed


def build_parser() -> argparse.ArgumentParser:
    """English: Expose every segmentation setting as a flag.
    Hindi: Har segmentation setting ko command-line flag banata hai.
    """
    parser = argparse.ArgumentParser(description="Train the U-Net segmenter")
    parser.add_argument("--epochs", type=int, default=SEGMENTER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=SEGMENTER_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=SEGMENTER_LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience in epochs. 0 disables it.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.28,
        help=(
            "Background patches to add per positive patch (~0.25-0.30). "
            "Hindi: Background patches ka ratio."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test mode: 2 epochs, 16 patches, saving to _smoketest_archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite existing best checkpoints even if new metric is lower.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from config import PROJECT_ROOT, RANDOM_SEED

    set_seed(args.seed if args.seed is not None else RANDOM_SEED)

    splits = load_split()
    train_dataset = SegmentationPatchDataset(
        splits["train"], train=True, negative_ratio=float(args.negative_ratio)
    )
    val_dataset = SegmentationPatchDataset(splits["val"], train=False)

    epochs = args.epochs
    model_dir = None
    metrics_dir = None
    curves_dir = None

    if args.smoke:
        print("\n" + "=" * 60)
        print(">>> [SMOKE TEST MODE] Running segmenter with 16 patches, 2 epochs.")
        print(">>> Smoke outputs will be saved to _smoketest_archive directories.")
        print("=" * 60 + "\n")
        train_dataset.records = train_dataset.records[:16]
        val_dataset.records = val_dataset.records[:16]
        epochs = 2
        model_dir = PROJECT_ROOT / "models" / "_smoketest_archive"
        metrics_dir = PROJECT_ROOT / "results" / "_smoketest_archive"
        curves_dir = PROJECT_ROOT / "results" / "_smoketest_archive" / "training_curves"

    print(f"Device: {DEVICE}")
    print("English: Segmentation patch counts")
    print("Hindi: Segmentation patch counts")
    print(f"  train: {len(train_dataset)}")
    print(f"  val  : {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit(
            "No usable segmentation patches found. Images with a missing mask "
            "or an entirely black mask are excluded by design.\n"
            "Hindi: Koi usable segmentation patch nahi mila. Jin images ka mask "
            "nahi hai ya poora black hai wo by design exclude hote hain."
        )

    train_segmenter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=DEVICE,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        patience=args.patience,
        model_dir=model_dir,
        metrics_dir=metrics_dir,
        curves_dir=curves_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
