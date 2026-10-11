"""English: CLI entry point for training the two-stage classifiers.
Hindi: Two-stage classifiers train karne ke liye CLI entry point.
"""

import argparse

from config import (
    BATCH_SIZE,
    DEFECT_PATCH_THRESHOLD,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_EPOCHS,
    FINETUNE_UNFREEZE,
    NUM_WORKERS,
)
from dataset import PatchDataset, load_split
from training import STAGE_SPEC, set_seed, train


def build_parser() -> argparse.ArgumentParser:
    """English: Define every training knob as a command-line flag.
    Hindi: Har training setting ko command-line flag banata hai.

    English: The requirement is that changing epochs or a learning rate never
    requires editing source code, so every value that a student might want to
    try is exposed here rather than hard-coded.
    Hindi: Requirement ye hai ki epochs ya learning rate badalne ke liye source
    code edit na karna pade, isliye har value yahan flag banaya gaya hai.
    """
    parser = argparse.ArgumentParser(
        description="Train a fabric defect classifier (two-stage pipeline)"
    )
    parser.add_argument(
        "--stage",
        choices=["binary", "defect_type"],
        required=True,
        help=(
            "binary = Normal vs Defective. defect_type = the 12 defect types. "
            "Hindi: binary = Normal/Defective, defect_type = 12 defect types."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "efficientnet"],
        default="cnn",
        help="cnn = custom baseline, efficientnet = transfer learning",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience in epochs. 0 disables it.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable the weighted sampler (ablation only).",
    )
    parser.add_argument(
        "--defect-threshold",
        type=float,
        default=DEFECT_PATCH_THRESHOLD,
        help=(
            "Mask fraction above which a patch counts as defective. "
            "Hindi: Mask fraction jisse zyada ho tab patch defective gino."
        ),
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=None,
        help="Unfreeze the EfficientNet tail for a second phase.",
    )
    parser.add_argument(
        "--no-finetune",
        dest="finetune",
        action="store_false",
        help="Train only the classifier head.",
    )
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
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
    parser = build_parser()
    args = parser.parse_args()

    from config import PROJECT_ROOT, RANDOM_SEED

    seed = args.seed if args.seed is not None else RANDOM_SEED
    set_seed(seed)

    spec = STAGE_SPEC[args.stage]
    epochs = args.epochs if args.epochs is not None else spec["epochs"]
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else spec["learning_rate"]
    )

    finetune = args.finetune
    if finetune is None:
        finetune = FINETUNE_UNFREEZE and args.model == "efficientnet"

    splits = load_split()
    train_dataset = PatchDataset(
        splits["train"], stage=args.stage, train=True, threshold=args.defect_threshold
    )
    val_dataset = PatchDataset(
        splits["val"], stage=args.stage, train=False, threshold=args.defect_threshold
    )

    model_dir = None
    metrics_dir = None
    curves_dir = None
    finetune_epochs = args.finetune_epochs

    if args.smoke:
        print("\n" + "=" * 60)
        print(">>> [SMOKE TEST MODE] Running with 16 patches, 2 epochs.")
        print(">>> Smoke outputs will be saved to _smoketest_archive directories.")
        print("=" * 60 + "\n")
        train_dataset.records = train_dataset.records[:16]
        val_dataset.records = val_dataset.records[:16]
        epochs = 2
        finetune_epochs = 1
        model_dir = PROJECT_ROOT / "models" / "_smoketest_archive"
        metrics_dir = PROJECT_ROOT / "results" / "_smoketest_archive"
        curves_dir = PROJECT_ROOT / "results" / "_smoketest_archive" / "training_curves"

    print(f"Device: {DEVICE}")
    print(f"Stage: {args.stage}  Model: {args.model}  Seed: {seed}")
    print(f"Defect patch threshold: {args.defect_threshold}")

    print("\nEnglish: Patch distribution")
    print("Hindi: Patch distribution")
    print(f"  train ({len(train_dataset)} patches)")
    for name, count in train_dataset.distribution().items():
        print(f"    {name:<16} {count}")
    if args.stage == "binary":
        print(f"  patch sources: {train_dataset.source_distribution()}")
        print(
            "  note: 'hard_negative' patches come from defective strips but "
            "contain no defect."
        )
    print(f"  val ({len(val_dataset)} patches)")
    for name, count in val_dataset.distribution().items():
        print(f"    {name:<16} {count}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit(
            "Empty dataset for this stage. Check --defect-threshold.\n"
            "Hindi: Dataset khaali hai. --defect-threshold check karein."
        )

    metrics = train(
        stage=args.stage,
        model_name=args.model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=DEVICE,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        num_workers=args.num_workers,
        finetune=finetune,
        finetune_epochs=finetune_epochs,
        patience=args.patience,
        balance=not args.no_balance,
        threshold=args.defect_threshold,
        model_dir=model_dir,
        metrics_dir=metrics_dir,
        curves_dir=curves_dir,
        overwrite=args.overwrite,
    )

    print(f"\nCompleted stage={args.stage} model={args.model}")
    print(
        "Note: these numbers come from the run above only. No result has been "
        "assumed or copied."
    )


if __name__ == "__main__":
    main()
