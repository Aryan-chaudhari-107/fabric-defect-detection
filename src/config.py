"""English: Central configuration for the fabric defect detection project.
Hindi: Project ki important settings ek hi jagah rakhne ke liye ye file use hoti hai.
Isse code ke andar hard-coded paths aur values kam hote hain.
"""

from pathlib import Path

import torch

# English: Project root is two levels above this source file.
# Hindi: Project ka main folder automatically identify kiya ja raha hai.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# English: Raw AITEX dataset directories.
# Hindi: Ye wahi original folders hain jo humne download kiye hain.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFECT_DIR = RAW_DATA_DIR / "Defect_images"
NODEFECT_DIR = RAW_DATA_DIR / "NODefect_images"
MASK_DIR = RAW_DATA_DIR / "Mask_images"

# English: Split manifest produced by split_dataset.py.
# Hindi: split_dataset.py ye file banata hai. Isme original image ke paths hote hain.
SPLIT_FILE = PROJECT_ROOT / "data" / "split.json"

# English: Patch geometry.
# Hindi: AITEX images 256 x 4096 hain, isliye hum unhe chhote patches mein todte hain.
# English: 256 is divisible by every U-Net downsample factor (2, 4, 8, 16, 32),
# so a 256x256 patch can be segmented without any resizing or cropping.
# Hindi: 256 har U-Net downsampling factor (2,4,8,16,32) se divisible hai,
# isliye 256x256 patch ko resize ya crop kiye bina segment kiya ja sakta hai.
PATCH_SIZE = 256
PATCH_STRIDE = 256

# English: Split ratios and reproducibility.
# Hindi: Split ka proportion aur random seed. Seed fix karne se results repeatable rehte hain.
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# English: When True, split_dataset.py keeps every image from the same physical
# fabric roll inside a single split. Default is False because AITEX has only
# 8 distinct rolls, so grouping by roll would remove almost every class from
# validation and test. The limitation is documented instead of hidden.
# Hindi: True hone par split_dataset.py ek hi fabric roll ke saare images ek
# hi split mein rakhega. Default False hai kyunke AITEX mein sirf 8 rolls hain
# aur roll-wise split karne se validation/test se lagbhag saari classes hat
# jaati hain.
STRICT_GROUP_SPLIT = False

# English: Patch-level defect labelling threshold.
# Hindi: Patch ko "defective" label dene ke liye minimum mask fraction.
#
# English: Measured patch statistics for the training split are:
#   min = 0.000122, p10 = 0.000229, p25 = 0.000572, median = 0.007904
# A value of 0.0002 corresponds to roughly 13 defect pixels inside a 256x256
# patch (13 / 65536). Anything above the minimum non-zero ratio is a real
# annotated defect rather than antialiasing, and anything below ~10 pixels is
# too small to be a reliable target. This threshold is configurable from the
# command line via --defect-threshold.
# Hindi: Training split ke patch statistics: min=0.000122, p10=0.000229,
# median=0.007904. 0.0002 ka matlab hai 256x256 patch mein lagbhag 13 defect
# pixels. Ye command line se bhi badla ja sakta hai.
DEFECT_PATCH_THRESHOLD = 0.0002

# English: Segmentation patches are kept whenever they contain at least this
# many defect pixels, independent of the classifier threshold.
# Hindi: Segmentation patches tab rakhe jaate hain jab unme kam se kam itne
# defect pixels hon. Ye classifier threshold se alag hai.
SEG_MIN_DEFECT_PIXELS = 8

# English: Caps that stop easy background patches from swamping the binary
# classifier. Documented in README under "Class imbalance".
# Hindi: Ye limits binary classifier ko background patches se bharne se
# rokhti hain.
MAX_NORMAL_PATCHES_PER_IMAGE = 4
MAX_HARD_NEGATIVES_PER_DEFECT_IMAGE = 2

# English: Training hyper-parameters.
# Hindi: Training ke dauran use hone wale settings.
BINARY_EPOCHS = 20
DEFECT_TYPE_EPOCHS = 20
BATCH_SIZE = 16
SEGMENTER_BATCH_SIZE = 8

BINARY_LEARNING_RATE = 1e-3
DEFECT_TYPE_LEARNING_RATE = 3e-4
FINETUNE_LEARNING_RATE = 3e-5

SEGMENTER_EPOCHS = 20
SEGMENTER_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# English: Early stopping.
# Hindi: Early stopping tab chalti hai jab validation metric improve na ho.
EARLY_STOPPING_PATIENCE = 5

# English: Transfer learning phases for EfficientNet-B0.
# Hindi: EfficientNet-B0 ke liye do phases: pehle sirf head train, phir
# backbone ka hissa unfreeze karke fine-tune.
FINETUNE_EPOCHS = 10
FINETUNE_UNFREEZE_FROM_BLOCK = 5
FINETUNE_UNFREEZE = True

# English: Runtime settings.
# Hindi: Training chalane ke liye device aur workers.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0

# English: Image read cache size. Patches are read lazily from the original
# strip, so caching avoids decoding the same 256x4096 PNG once per patch.
# Hindi: Patch lene ke liye original strip dobara padha jata hai, isliye cache
# se ek hi image baar baar decode nahi hoti.
IMAGE_CACHE_SIZE = 8

# English: Saved model locations.
# Hindi: Trained models yahan save honge.
CLASSIFIER_MODEL_DIR = PROJECT_ROOT / "models" / "classifier"
SEGMENTER_MODEL_DIR = PROJECT_ROOT / "models" / "segmenter"

# English: Results and report output locations.
# Hindi: Metrics, graphs aur visualizations yahan save honge.
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"
TRAINING_CURVES_DIR = RESULTS_DIR / "training_curves"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOCALIZATION_DIR = RESULTS_DIR / "localization"
LOGS_DIR = RESULTS_DIR / "logs"
BACKUP_MODEL_DIR = PROJECT_ROOT / "models" / "_backup"
BACKUP_RESULTS_DIR = RESULTS_DIR / "_backup"

# English: Cap on qualitative localization figures written per evaluation run.
# Hindi: Har evaluation run mein kitne localization figures banein, uski limit.
MAX_LOCALIZATION_FIGURES = 6

# English: Inference thresholds for the two-stage pipeline.
# Hindi: Two-stage pipeline ke inference thresholds.
STAGE1_DEFECT_PROB_THRESHOLD = 0.5
STAGE1_MIN_DEFECTIVE_PATCHES = 1
SEGMENTATION_THRESHOLD = 0.5

for _directory in (
    CLASSIFIER_MODEL_DIR,
    SEGMENTER_MODEL_DIR,
    METRICS_DIR,
    CONFUSION_MATRIX_DIR,
    TRAINING_CURVES_DIR,
    PREDICTIONS_DIR,
    LOCALIZATION_DIR,
    LOGS_DIR,
    BACKUP_MODEL_DIR,
    BACKUP_RESULTS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)


def checkpoint_path(stage: str, model_name: str, best: bool = True):
    """English: Resolve the checkpoint path for a stage and architecture.
    Hindi: Stage aur architecture ke liye checkpoint ka path deta hai.
    """
    suffix = "best" if best else "last"
    return CLASSIFIER_MODEL_DIR / f"{stage}_{model_name}_{suffix}.pt"
