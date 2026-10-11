"""English: AITEX defect-code definitions and two-stage label spaces.
Hindi: AITEX defect codes aur two-stage label spaces ka definition.
"""

from pathlib import Path
from typing import Dict, List, Optional

# English: Official AITEX defect codes.
# Hindi: AITEX ke official defect codes aur unke naam.
DEFECT_CLASSES = {
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

NORMAL_CODE = "000"
NORMAL_CLASS = "Normal"

# English: Stage 1 label space: binary normal versus defective.
# Hindi: Stage 1 ka label space: binary Normal versus Defective.
BINARY_CLASSES: List[str] = [NORMAL_CLASS, "Defective"]
BINARY_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(BINARY_CLASSES)}
BINARY_NORMAL = 0
BINARY_DEFECTIVE = 1

# English: Stage 2 label space: the 12 real defect types, Normal excluded.
# Hindi: Stage 2 ka label space: 12 asli defect types. 'Normal' ismein nahi hai
# kyunki Normal patches Stage 1 mein hi filter ho jaate hain.
DEFECT_TYPE_CLASSES: List[str] = [
    name for name in DEFECT_CLASSES.values() if name != NORMAL_CLASS
]
DEFECT_TYPE_TO_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(DEFECT_TYPE_CLASSES)
}
DEFECT_TYPE_INDEX_TO_CLASS: Dict[int, str] = {
    i: name for name, i in DEFECT_TYPE_TO_INDEX.items()
}
NUM_DEFECT_TYPES = len(DEFECT_TYPE_CLASSES)

# English: Classes with exactly one original image. They can never be split
# across train, validation and test, so they have no meaningful test score.
# Hindi: Jin classes ka sirf ek original image hai. Inka test score meaningful
# nahi ho sakta, isliye evaluation mein ye alag se report hote hain.
SINGLE_IMAGE_CLASS_NAMES = {"Knots", "Contamination", "Weft Crack"}


def parse_filename(path: Path) -> Optional[dict]:
    """English: Split an AITEX filename into its parts.
    Hindi: AITEX filename ke hisse alag karta hai.

    Expected format is ``nnnn_ddd_ff`` where ``nnnn`` is the image number,
    ``ddd`` is the defect code and ``ff`` is the fabric (roll) code.

    One file in our copy of the dataset (``0018_00_01.png``) uses a two digit
    defect code, so the defect code is zero-padded before lookup.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    image_number, defect_code, fabric_code = parts[0], parts[1], parts[2]
    if not image_number.isdigit():
        return None
    return {
        "image_number": image_number,
        "defect_code": defect_code.zfill(3),
        "raw_defect_code": defect_code,
        "fabric_code": fabric_code,
    }


def code_to_class(code: str) -> Optional[str]:
    """English: Convert a defect code to a class name.
    Hindi: Defect code ko class name mein convert karta hai.
    Unknown codes return ``None`` so that bad data is never silently mapped
    to the Normal class.
    """
    return DEFECT_CLASSES.get(code.zfill(3))


def class_of_image(path: Path) -> Optional[str]:
    """English: Determine the class of an image from its filename.
    Hindi: Filename se image ki class decide karta hai.
    """
    parsed = parse_filename(path)
    if parsed is None:
        return None
    return code_to_class(parsed["defect_code"])


def fabric_code_of_image(path: Path) -> Optional[str]:
    """English: Return the fabric roll code of an image.
    Hindi: Image ka fabric roll code return karta hai.

    English: The two-digit ``ff`` group is the physical fabric code. All normal
    images of one roll live in a single folder named after that roll, and the
    defect images carry the same two-digit code. Grouping by this code is what
    ``STRICT_GROUP_SPLIT`` uses to prevent near-duplicate strips from landing
    in different splits.
    Hindi: Do-digit ``ff`` group hi physical fabric code hai. Ek roll ke saare
    normal images usi roll ke folder mein hote hain aur defect images mein bhi
    wahi code hota hai. STRICT_GROUP_SPLIT isi code se group karta hai.
    """
    parsed = parse_filename(path)
    if parsed is None:
        return None
    return parsed["fabric_code"]


def is_normal_image(path: Path) -> bool:
    """English: Return True if the filename encodes defect code 000.
    Hindi: Filename mein defect code 000 hai to True return karta hai.
    """
    parsed = parse_filename(path)
    if parsed is None:
        return False
    return parsed["defect_code"] == NORMAL_CODE
