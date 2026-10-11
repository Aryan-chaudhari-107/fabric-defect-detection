"""English: General-purpose helper functions.
Hindi: Common image/file operations ko reusable banane ke liye ye utilities hain.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DEFECT_DIR,
    IMAGE_CACHE_SIZE,
    MASK_DIR,
    NODEFECT_DIR,
    PATCH_SIZE,
    RAW_DATA_DIR,
)


def validate_dataset_layout() -> None:
    """English: Fail early with an actionable message if the dataset is missing.
    Hindi: Agar dataset folders missing hain to clear error message deta hai.

    English: Raising a readable error here is much better than letting a
    ``FileNotFoundError`` surface 200 lines deep inside a training loop.
    Hindi: Yahan readable error dena behtar hai, warna training loop ke andar
    hi cryptic FileNotFoundError aayega.
    """
    missing = [
        str(path)
        for path in (RAW_DATA_DIR, DEFECT_DIR, NODEFECT_DIR, MASK_DIR)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Could not find the following expected dataset folders:\n  "
            + "\n  ".join(missing)
            + "\n\nHindi: AITEX dataset ko aise rakhein:\n"
            "  data/raw/Defect_images/\n  data/raw/NODefect_images/\n"
            "  data/raw/Mask_images/"
        )


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _read_image_cached(path_str: str) -> np.ndarray:
    """English: Read an image from disk with a small LRU cache.
    Hindi: Image ko disk se padhta hai aur chhota LRU cache rakhta hai.

    English: Patches are cut lazily from the original strip, so without a cache
    each 256x256 patch would decode the same ~700 KB PNG again. The array is
    returned read-only so callers cannot corrupt the cache.
    Hindi: Patches original strip se lazily kaate hain, isliye bina cache ke
    har patch ke liye wahi ~700 KB PNG dobara decode hoga.
    """
    image = cv2.imread(path_str)
    if image is None:
        raise ValueError(
            f"Unable to read image: {path_str}\n"
            "Hindi: Image corrupt ho sakti hai ya path galat hai."
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return rgb


def read_image(path: Path) -> np.ndarray:
    """English: Read an image using OpenCV and return cached RGB data.
    Hindi: Image ko safely read karke RGB format mein return karta hai.
    """
    return _read_image_cached(str(path))


def clear_image_cache() -> None:
    """English: Drop cached image data.
    Hindi: Cached image data hata deta hai.
    """
    _read_image_cached.cache_clear()


def read_mask(path: Path) -> np.ndarray:
    """English: Read a mask as a binary image.
    Hindi: Mask ko black/white binary form mein convert karta hai.

    AITEX masks are saved as antialiased grayscale PNGs, so they contain many
    intermediate grey values instead of only 0 and 255. Thresholding at 127
    is therefore required; a ``> 0`` test would label almost the whole strip
    as defective.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(
            f"Unable to read mask: {path}\n"
            "Hindi: Mask corrupt ho sakta hai ya path galat hai."
        )
    return (mask > 127).astype(np.uint8)


def find_mask(image_path: Path, mask_dir: Path = MASK_DIR) -> Optional[List[Path]]:
    """English: Locate every mask belonging to an image.
    Hindi: Image ke saare mask files dhoondta hai.

    AITEX uses two mask layouts, and our copy contains both:

    * ``nnnn_ddd_ff_mask.png``  - one mask
    * ``nnnn_ddd_ff_mask1.png`` and ``_mask2.png`` - two separate masks
      (``0044_019_04`` and ``0097_030_03`` in this dataset)

    Returning a list lets the caller merge multiple masks instead of silently
    dropping one defect region.

    Returns ``None`` when no mask exists. One defect image, ``0100_025_08``,
    has no mask at all and is therefore classification-only.
    """
    stem = image_path.stem
    candidates = sorted(Path(mask_dir).glob(f"{stem}_mask*.png"))
    return candidates or None


def merged_mask(
    image_path: Path,
    mask_dir: Path = MASK_DIR,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """English: Load and merge all masks of an image into one binary mask.
    Hindi: Image ke saare masks ko ek binary mask mein merge karta hai.

    English: ``target_shape`` should normally be the shape of the source image.
    Every mask in this dataset is either 4096 wide or, for ``0094_027_05``,
    3796 wide, matching its image. Merging two masks of differing widths would
    broadcast-fail, and a zero-filled merge anchored to the image shape is the
    correct interpretation for a narrower mask.

    Returns ``None`` when the image has no mask or when the merged mask has
    zero defect pixels. ``0100_025_08`` has no mask at all and
    ``0106_010_03`` has an entirely black mask; both return ``None`` and are
    therefore excluded from segmentation.
    """
    paths = find_mask(image_path, mask_dir)
    if not paths:
        return None

    if target_shape is None:
        first = read_mask(paths[0])
        height, width = first.shape[:2]
    else:
        height, width = target_shape

    combined = np.zeros((height, width), dtype=np.uint8)

    for path in paths:
        current = read_mask(path)
        rows = min(height, current.shape[0])
        cols = min(width, current.shape[1])
        combined[:rows, :cols] = np.logical_or(
            combined[:rows, :cols], current[:rows, :cols]
        ).astype(np.uint8)

    if combined.sum() == 0:
        return None
    return combined


def pad_to_patch_grid(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """English: Pad an image so both dimensions are multiples of ``patch_size``.
    Hindi: Image ke dono sides ko ``patch_size`` ka multiple bana deta hai.

    English: Edge replication is used so the padded strip keeps the fabric
    texture at the border instead of introducing a black band that the model
    would learn to treat as an artefact.
    Hindi: Edge replication use karte hain taaki padding ke border par fabric
    ka texture bana rahe, black band na aaye.
    """
    height, width = image.shape[:2]
    pad_y = (patch_size - height % patch_size) % patch_size
    pad_x = (patch_size - width % patch_size) % patch_size
    if pad_y == 0 and pad_x == 0:
        # English: ``cv2.copyMakeBorder`` is needed even when no padding is
        # required, because ``cv2.copyMakeBorder`` on an empty border still
        # returns a fresh writable array, while a no-op would return a
        # read-only view into the cache.
        return np.array(image)
    return cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)


def iter_patch_origins(
    width: int,
    height: int,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_SIZE,
) -> List[Tuple[int, int]]:
    """English: Deterministic list of patch top-left coordinates.
    Hindi: Patch ke top-left coordinates ki deterministic list.

    English: The final patch is always included even when it extends past the
    original width. ``0094_027_05.png`` is 3796 px wide, so dropping the
    remainder would discard its defect region entirely.
    Hindi: Last patch hamesha include hota hai, chahe wo original width se
    aage jaye. 3796 px wide image ka remainder drop karne se defect region
    poora chala jayega.
    """
    origins: List[Tuple[int, int]] = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            origins.append((x, y))
            x += stride
        y += stride
    return origins


def slice_patch(
    padded: np.ndarray,
    x: int,
    y: int,
    patch_size: int = PATCH_SIZE,
    copy: bool = False,
) -> np.ndarray:
    """English: Cut one patch from an already padded array.
    Hindi: Padded array se ek patch kaat ta hai.

    English: ``copy=True`` is required whenever the patch is handed to
    ``torchvision.transforms.ToTensor``. The image cache returns read-only
    arrays, and ``ToTensor`` raises a warning and would otherwise hand PyTorch
    a buffer it must not modify.
    Hindi: Patch ko ``ToTensor`` ko dene se pehle ``copy=True`` chahiye, kyunki
    image cache read-only arrays deta hai aur PyTorch ko modify karne se
    rokna zaroori hai.
    """
    patch = padded[y:y + patch_size, x:x + patch_size]
    if copy:
        return np.array(patch)
    return patch


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """English: Blend a binary mask over an RGB image for visualisation.
    Hindi: Defect region ko highlight karke visualization banata hai.
    """
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    output = image.astype(np.float32).copy()
    tint = np.zeros_like(output)
    tint[:, :] = color
    selection = mask.astype(bool)
    output[selection] = output[selection] * (1 - alpha) + tint[selection] * alpha
    return output.astype(np.uint8)


def mask_bounding_box(mask: np.ndarray) -> Optional[Dict[str, int]]:
    """English: Smallest axis-aligned box that covers the predicted mask.
    Hindi: Predicted mask ko gherne wala sabse chhota bounding box.

    English: This is a derived visualisation of a segmentation result, not an
    object-detection output. The project is not doing detection.
    Hindi: Ye segmentation result ka derived visualization hai, object
    detection ka output nahi. Ye project detection nahi kar raha.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
        "width": int(x_max - x_min + 1),
        "height": int(y_max - y_min + 1),
    }
