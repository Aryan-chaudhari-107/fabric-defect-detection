"""English: Patch datasets for the two-stage classifier and the segmenter.
Hindi: Two-stage classifier aur segmenter ke liye patch datasets.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    DEFECT_PATCH_THRESHOLD,
    MASK_DIR,
    MAX_HARD_NEGATIVES_PER_DEFECT_IMAGE,
    MAX_NORMAL_PATCHES_PER_IMAGE,
    PATCH_SIZE,
    PATCH_STRIDE,
    PROJECT_ROOT,
    SEG_MIN_DEFECT_PIXELS,
    SPLIT_FILE,
)
from labels import (
    BINARY_CLASSES,
    BINARY_NORMAL,
    DEFECT_TYPE_TO_INDEX,
    NORMAL_CLASS,
    SINGLE_IMAGE_CLASS_NAMES,
    class_of_image,
)
from utils import (
    iter_patch_origins,
    merged_mask,
    pad_to_patch_grid,
    read_image,
    slice_patch,
)


def load_split(split_name: str = "splits") -> Dict[str, List[Path]]:
    """English: Read the image-level split manifest.
    Hindi: split.json se image-level split load karta hai.

    English: Raises a readable error if the manifest is missing, because that
    usually means ``split_dataset.py`` has not been run yet.
    Hindi: Manifest na hone par readable error deta hai, aam taur par iska
    matlab hai split_dataset.py abhi nahi chalaya gaya.
    """
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {SPLIT_FILE}.\n"
            "Hindi: Pehle `python src/split_dataset.py` chalayein. "
            "Original images ka split isi se banta hai."
        )

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        name: [PROJECT_ROOT / relative for relative in paths]
        for name, paths in data["splits"].items()
    }


def build_patch_records(
    image_paths: List[Path],
    threshold: float = DEFECT_PATCH_THRESHOLD,
    include_hard_negatives: bool = True,
) -> List[dict]:
    """English: Build a deterministic patch index for a list of images.
    Hindi: Image list se deterministic patch index banata hai.

    English: For every original image the mask (if any) is merged and padded
    with exactly the same geometry as the image, so a patch and its mask are
    always pixel-aligned. Patches are emitted in a fixed order, so patch
    generation is fully deterministic.

    Label policy:

    * a patch from a normal image is always Normal
    * a patch from a defective image is Defective when its mask fraction
      reaches ``threshold``
    * the remaining background patches of a defective image are hard
      negatives: they are the hardest kind of Normal sample because the same
      fabric roll and lighting are present but the defect is not. They are
      capped per image so they cannot dominate the binary task.
    * ``0100_025_08`` has no mask, so all of its patches are hard negatives
      for the binary task and it is excluded from stage 2 and segmentation

    Hindi: Har image ka mask merge karke image ke bilkul same geometry se pad
    kiya jata hai, isliye patch aur mask hamesha aligned rehte hain.
    """
    records: List[dict] = []

    for image_path in image_paths:
        class_name = class_of_image(image_path)
        if class_name is None:
            continue

        image = read_image(image_path)
        height, width = image.shape[:2]
        padded_image = pad_to_patch_grid(image, PATCH_SIZE)

        is_defective_image = class_name != NORMAL_CLASS
        mask = (
            merged_mask(image_path, MASK_DIR, target_shape=(height, width))
            if is_defective_image
            else None
        )
        padded_mask = pad_to_patch_grid(mask, PATCH_SIZE) if mask is not None else None

        origins = iter_patch_origins(width, height, PATCH_SIZE, PATCH_STRIDE)

        if not is_defective_image:
            # English: Sample a fixed number of normal patches per image. Using
            # all 16 patches from 99 normal images produced 1584 easy negatives
            # against only 119 defective patches, which made the binary task
            # trivially imbalanced.
            # Hindi: Har normal image se fixed number ke patches lete hain.
            # Saare 16 lene se 1584 easy negatives ban jaate the jo sirf 119
            # defective patches ke against bahut zyada the.
            step = max(1, len(origins) // MAX_NORMAL_PATCHES_PER_IMAGE)
            for index, (x, y) in enumerate(origins):
                if index % step:
                    continue
                records.append(
                    {
                        "image_path": image_path,
                        "x": x,
                        "y": y,
                        "class_name": NORMAL_CLASS,
                        "binary_label": BINARY_NORMAL,
                        "defect_type_index": None,
                        "source": "normal_image",
                    }
                )
            continue

        # English: Defective image: separate true defective patches from
        # background patches, then cap the background ones.
        # Hindi: Defective image: asli defective patches aur background patches
        # alag karke background ko limit karte hain.
        defect_records: List[dict] = []
        background_origins: List[Tuple[int, int]] = []

        for x, y in origins:
            patch_mask = (
                slice_patch(padded_mask, x, y) if padded_mask is not None else None
            )
            fraction = float(patch_mask.mean()) if patch_mask is not None else 0.0

            if fraction >= threshold:
                defect_records.append(
                    {
                        "image_path": image_path,
                        "x": x,
                        "y": y,
                        "class_name": class_name,
                        "binary_label": 1,
                        "defect_type_index": DEFECT_TYPE_TO_INDEX[class_name],
                        "mask": patch_mask,
                        "mask_fraction": fraction,
                        "source": "defect",
                    }
                )
            else:
                background_origins.append((x, y))

        records.extend(defect_records)

        if include_hard_negatives and background_origins:
            step = max(1, len(background_origins) // MAX_HARD_NEGATIVES_PER_DEFECT_IMAGE)
            for index, (x, y) in enumerate(background_origins):
                if index % step:
                    continue
                records.append(
                    {
                        "image_path": image_path,
                        "x": x,
                        "y": y,
                        "class_name": NORMAL_CLASS,
                        "binary_label": BINARY_NORMAL,
                        "defect_type_index": None,
                        "source": "hard_negative",
                    }
                )

    for position, record in enumerate(records):
        record["index"] = position

    return records


def classification_transform():
    """English: Normalisation transform for classifier inputs.
    Hindi: Classifier input ke liye normalisation transform.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def augment_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """English: Random flip and rotation on an already normalised tensor.
    Hindi: Already normalised tensor par random flips aur rotation lagata hai.

    English: ``torchvision.transforms.RandomHorizontalFlip`` dispatches to PIL
    when given a numpy array, so augmentation is applied on tensors instead.
    An arbitrary rotation angle is deliberately avoided here because
    fabric textures are strongly anisotropic; small-angle flips and 90 degree
    rotations keep the weave orientation consistent.
    Hindi: Numpy array dene par torchvision PIL use karta hai, isliye
    augmentation tensor par kiya jata hai.
    """
    if torch.rand(1).item() < 0.5:
        tensor = torch.flip(tensor, dims=[2])
    if torch.rand(1).item() < 0.5:
        tensor = torch.flip(tensor, dims=[1])
    return tensor


def _load_patch_image(record: dict, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """English: Read one patch from its original strip.
    Hindi: Original strip se ek patch padhta hai.

    English: The patch is copied because the image cache is read-only and
    ``ToTensor`` needs a writable buffer.
    Hindi: Patch copy kiya jata hai kyunki image cache read-only hai aur
    ``ToTensor`` ko writable buffer chahiye.
    """
    image = read_image(record["image_path"])
    padded = pad_to_patch_grid(image, patch_size)
    return slice_patch(padded, record["x"], record["y"], patch_size, copy=True)


class PatchDataset(Dataset):
    """English: Shared patch dataset for both classification stages.
    Hindi: Dono classification stages ke liye shared patch dataset.

    English: ``stage='binary'`` yields (image, 0/1). ``stage='defect_type'``
    yields (image, 0..11) and keeps only defective patches.
    Hindi: ``stage='binary'`` se (image, 0/1) milta hai. ``stage='defect_type'``
    se (image, 0..11) milta hai aur sirf defective patches rehte hain.
    """

    def __init__(
        self,
        image_paths: List[Path],
        stage: str = "binary",
        train: bool = False,
        threshold: float = DEFECT_PATCH_THRESHOLD,
    ):
        if stage not in {"binary", "defect_type"}:
            raise ValueError(f"Unknown stage: {stage}")

        self.stage = stage
        self.records = build_patch_records(image_paths, threshold=threshold)

        if stage == "defect_type":
            self.records = [
                record
                for record in self.records
                if record["defect_type_index"] is not None
            ]

        self.transform = classification_transform()
        self.augment = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> Tuple[torch.Tensor, int]:
        record = self.records[position]
        patch = _load_patch_image(record)
        tensor = self.transform(patch)
        if self.augment:
            tensor = augment_tensor(tensor)

        label = (
            record["binary_label"]
            if self.stage == "binary"
            else int(record["defect_type_index"])
        )
        return tensor, label

    def distribution(self) -> Dict[str, int]:
        """English: Count patches per class in the space this stage uses.
        Hindi: Har class ke patches ki sankhya, us stage ke label space mein.

        English: For the binary stage this reports Normal versus Defective. The
        defect type of a defective patch is deliberately not reported here,
        because Stage 1 is not trained to distinguish types and listing 12
        defect names there would misrepresent what the model is being asked to
        learn.
        Hindi: Binary stage ke liye ye Normal aur Defective ki sankhya batata
        hai. Defective patch ka defect type yahan jaan-boojh kar nahi dikhaya
        jata, kyunki Stage 1 ko types alag karna nahi seekhaya ja raha.
        """
        if self.stage == "binary":
            counter = Counter(
                BINARY_CLASSES[record["binary_label"]]
                for record in self.records
            )
            return dict(sorted(counter.items()))

        counter = Counter(record["class_name"] for record in self.records)
        return dict(sorted(counter.items()))

    def source_distribution(self) -> Dict[str, int]:
        """English: Count patches by origin (normal image vs hard negative).
        Hindi: Patches ki sankhya origin ke hisaab se (normal image ya hard negative).
        """
        counter = Counter(record["source"] for record in self.records)
        return dict(sorted(counter.items()))

    def binary_label_distribution(self) -> Dict[str, int]:
        """English: Normal versus Defective patch counts.
        Hindi: Normal aur Defective patches ki sankhya.
        """
        counter = Counter(
            BINARY_CLASSES[record["binary_label"]] for record in self.records
        )
        return dict(sorted(counter.items()))

    def missing_single_image_classes(self) -> List[str]:
        """English: Which one-image classes are absent from this split.
        Hindi: Kaun se ek-image classes is split mein missing hain.
        """
        present = set(self.distribution())
        return sorted(SINGLE_IMAGE_CLASS_NAMES - present)


class SegmentationPatchDataset(Dataset):
    """English: Patch-level segmentation dataset built only from real masks.
    Hindi: Segmentation dataset. Ye sirf un patches se banta hai jinke mask
    mein actual defect pixels hain.
    """

    def __init__(
        self,
        image_paths: List[Path],
        train: bool = False,
        min_defect_pixels: int = SEG_MIN_DEFECT_PIXELS,
        negative_ratio: float = 0.0,
    ):
        self.records = [
            record
            for record in build_patch_records(image_paths, threshold=0.0)
            if record.get("mask") is not None
            and int(record["mask"].sum()) >= min_defect_pixels
        ]

        self.train = train
        self.transform = classification_transform()

        # English: Add clean patches and hard negatives (~25-30% of training data)
        # to teach the model true background and reduce false positives.
        # Hindi: Background aur hard negative patches add karta hai taaki model false positives na de.
        if negative_ratio > 0.0 and self.records:
            wanted_count = max(1, int(len(self.records) * negative_ratio / max(1e-5, (1.0 - negative_ratio))))
            self.records = self.records + self._build_negatives(
                image_paths, wanted_count
            )

    def _build_negatives(
        self, image_paths: List[Path], wanted_count: int
    ) -> List[dict]:
        """English: Collect clean normal patches and hard negatives for segmentation.
        Hindi: Segmentation ke liye clean normal patches aur hard negatives collect karta hai.
        """
        negatives: List[dict] = []
        normal_paths = [p for p in image_paths if class_of_image(p) == NORMAL_CLASS]
        defect_paths = [p for p in image_paths if class_of_image(p) != NORMAL_CLASS]

        # 1. Hard negatives from defect images
        for image_path in defect_paths:
            mask = merged_mask(image_path, MASK_DIR)
            if mask is None:
                continue

            image = read_image(image_path)
            height, width = image.shape[:2]
            padded_mask = pad_to_patch_grid(mask, PATCH_SIZE)

            for x, y in iter_patch_origins(width, height, PATCH_SIZE, PATCH_STRIDE):
                if padded_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE].any():
                    continue
                negatives.append(
                    {
                        "image_path": image_path,
                        "x": x,
                        "y": y,
                        "mask": np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8),
                        "source": "hard_negative",
                    }
                )
                if len(negatives) >= wanted_count // 2:
                    break
            if len(negatives) >= wanted_count // 2:
                break

        # 2. Clean patches from normal images
        for image_path in normal_paths:
            image = read_image(image_path)
            height, width = image.shape[:2]
            for x, y in iter_patch_origins(width, height, PATCH_SIZE, PATCH_STRIDE):
                negatives.append(
                    {
                        "image_path": image_path,
                        "x": x,
                        "y": y,
                        "mask": np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8),
                        "source": "clean_normal",
                    }
                )
                if len(negatives) >= wanted_count:
                    return negatives
        return negatives[:wanted_count]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[position]
        patch = _load_patch_image(record)
        mask = record["mask"]

        if self.train:
            # English: Flips must be applied to image and mask together to keep
            # them aligned. No rotation is used for the same reason as the
            # classifier.
            # Hindi: Image aur mask dono par same flip lagana zaroori hai,
            # warna alignment bigad jayegi.
            if np.random.rand() < 0.5:
                patch = patch[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                patch = patch[::-1, :].copy()
                mask = mask[::-1, :].copy()

        image_tensor = self.transform(patch)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
        return image_tensor, mask_tensor.unsqueeze(0)
