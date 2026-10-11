# Fabric Defect Detection Using Deep Learning

Academic research prototype for automated fabric defect inspection using the
AITEX Fabric Image Database.

This is a research prototype. It is **not** a certified industrial quality
control system.

---

## 1. Objective

English:
Detect whether a fabric image contains a defect, classify the defect type,
localize the defective region, and expose the whole thing through a Streamlit
interface.

Hindi:
Fabric image mein defect hai ya nahi ye detect karna, defect ka type classify
karna, defect ki location batana, aur poora system Streamlit interface ke
through dikhana.

---

## 2. Final architecture

```
AITEX RAW DATA (247 images)
        |
        v
Image-level split  (train 173 / val 37 / test 37)
        |
        v
256 x 256 aligned patches
        |
        +-----------------------------+
        |                             |
        v                             v
Stage 1                          Segmentation
Binary classifier                U-Net
Normal vs Defective              Defect localization
        |                             |
        v                             |
Only defective patches              |
        |                             |
        v                             |
Stage 2                              |
12-class defect classifier           |
        |                             |
        +-------------+---------------+
                      v
          Image-level aggregation
                      |
                      v
        Final result:
        Normal / Defective
        Defect type
        Confidence
        Localized defect mask
                      |
                      v
                 Streamlit
```

---

## 3. Why two-stage classification

English:
A single 13-class classifier has to learn 13 decisions from the same 106
defective images. Splitting the task means Stage 1 only has to answer an easy,
well-balanced question, and Stage 2 only has to discriminate between defect
types using patches that are already known to contain a defect.

Concretely, this matters because of how the data looks:

| | Stage 1 (binary) | Stage 2 (defect type) |
| --- | --- | --- |
| Train patches | 705 | 109 |
| Normal patches | 596 | 0 |
| Defective patches | 109 | 109 |

Stage 1 sees roughly balanced data and only has to learn "is there a defect in
this 256x256 crop". Stage 2 never sees a Normal patch, so it never wastes
capacity on the easy majority class.

Hindi:
Ek 13-class classifier ko 13 decisions ek hi 106 defective images se seekhni
padti hain. Do stage mein baat karne se Stage 1 ko sirf ek aasan aur balanced
sawal jawab karna hai, aur Stage 2 sirf un patches ke beech discriminate karta
hai jinmein pehle se pata hai defect hai.

---

## 4. Dataset audit

Measured directly from `data/raw`, not copied from the dataset documentation.

| Item | Count |
| --- | --- |
| Defect images | 106 |
| Normal images | 141 |
| Classification total | 247 |
| Mask files | 107, covering 105 unique images |
| Image size | 256 x 4096 (246 images), 256 x 3796 (`0094_027_05.png`) |

### Class distribution

| Code | Class | Images |
| --- | --- | --- |
| 000 | Normal | 141 |
| 002 | Broken End | 9 |
| 006 | Broken Yarn | 8 |
| 010 | Broken Pick | 10 |
| 016 | Weft Curling | 3 |
| 019 | Fuzzyball | 39 |
| 022 | Cut Selvage | 9 |
| 023 | Crease | 5 |
| 025 | Warp Ball | 6 |
| 027 | Knots | 1 |
| 029 | Contamination | 1 |
| 030 | Nep | 14 |
| 036 | Weft Crack | 1 |

### Dataset version difference

The official AITEX page describes 245 original images. This copy has 247. The
two extra files are genuine captures, not corruption:

- `NODefect_images/2608691-202020u/0018_00_01.png` uses a two-digit defect code
  (`00` instead of `000`). Verified not a byte-level or pixel-level duplicate of
  `0018_000_01.png`.
- `0094_027_05.png` is 3796 px wide, a differently cropped capture.

No MD5 duplicates exist anywhere in the dataset.

---

## 5. Known irregularities and how the code handles them

| # | Problem | Handling |
| --- | --- | --- |
| 1 | `0044_019_04` and `0097_030_03` each have two mask files (`_mask1`, `_mask2`) | `utils.find_mask` returns a list, `utils.merged_mask` merges with logical OR. Neither defect region is lost. |
| 2 | `0100_025_08` has no mask | Usable for both classification stages. Its patches become hard negatives for Stage 1. Excluded from Stage 2 and segmentation, because there is no ground truth to learn a defect type from. |
| 3 | `0106_010_03_mask.png` is completely black | `merged_mask` returns `None`, so it is excluded from segmentation training. |
| 4 | Masks are antialiased grayscale, not binary | `read_mask` thresholds at **127**. Thresholding at `> 0` marks roughly 70% of every strip as defective. |
| 5 | `0094_027_05.png` is 3796 px wide | `iter_patch_origins` always emits the final patch and `pad_to_patch_grid` pads it with edge replication. At inference the padded region is cropped away, so the returned mask is exactly 256 x 3796. |
| 6 | `Knots`, `Contamination`, `Weft Crack` have one image each | Kept entirely in train. Reported as `absent_classes` in evaluation rather than scored as 0. |
| 7 | Median defect area is 0.009% of the image | Drives patch size and the Dice + BCE loss. See below. |

---

## 6. Why patching is required

English:
AITEX images are 256 x 4096 strips, not square photographs. A 13-class model
would have to resize a 4096 x 256 strip to 224 x 224, shrinking each defect by
a factor of about 18 in both dimensions. Since the median defect occupies
0.009% of the strip, most defects would occupy well under one pixel after
resizing and become unrecoverable.

Cutting the strip into 16 patches of 256 x 256 keeps native resolution, so a
defect stays a defect. 256 is also divisible by every U-Net downsampling factor
(2, 4, 8, 16, 32), so patches segment with no resizing at all.

Measured patch statistics for the training split:

```
min = 0.000122   p10 = 0.000229   p25 = 0.000572
median = 0.007904   p75 = 0.020935   p90 = 0.257504
```

Hindi:
AITEX images 256 x 4096 lambi strips hain, square photos nahi. 224 x 224 mein
resize karne par har defect dono directions mein lagbhag 18 guna chhota ho
jaata. Median defect sirf 0.009% area ka hai, isliye resize ke baad defect
ek pixel se bhi chhota reh jaata hai aur recover nahi hota.

---

## 7. Patch labelling and class imbalance

### Defect patch threshold

`DEFECT_PATCH_THRESHOLD = 0.0002`, which is about 13 defect pixels inside a
256 x 256 patch (13 / 65536). Anything above the minimum observed non-zero ratio
(0.000122) is a real annotated defect rather than mask antialiasing, and
anything below roughly 10 pixels is too small to be a reliable target.

Change it without editing code:

```bash
python src/train_classifier.py --stage binary --model cnn --defect-threshold 0.0005
```

### Negative sampling

English:
Taking all 16 patches from all 99 normal training images produced **1584 easy
negative patches** against only 119 defective patches. That makes the binary task
trivially imbalanced and inflates patch-level accuracy without meaning anything.

Instead:

- each normal image contributes at most 4 patches (`MAX_NORMAL_PATCHES_PER_IMAGE`)
- each defective image contributes at most 2 background patches
  (`MAX_HARD_NEGATIVES_PER_DEFECT_IMAGE`)

Those background patches are *hard negatives*: same roll, same lighting, same
weave, but no defect. They are far more useful than clean normal-roll patches.

Result: 596 Normal vs 109 Defective in train, balanced further by a
`WeightedRandomSampler`.

Patch-level validation/test sets use the same per-image caps (4 normal patches, 2 hard negatives), so patch-level metrics are on a subsample; image-level evaluation scans all patches.

A weighted sampler is used and **not** a weighted loss. Doing both would
double-correct the imbalance and push the effective per-class weight to roughly
inverse-square, which overfits the smallest classes. `--no-balance` exists as
an ablation switch.

Hindi:
Saare normal images ke saare patches lene se 1584 easy negatives ban jaate the
jo sirf 119 defective patches ke against the. Isliye har normal image se max 4
patches aur har defective image se max 2 background patches lete hain. Ye
background patches hard negatives hain: wahi roll, wahi lighting, wahi weave,
par defect nahi.

### Known issue fixed: Binary Stage Sampler Bug

In earlier iterations, `make_balanced_sampler` counted patches by `record['class_name']`. In the binary stage, "Defective" was split into 12 defect-type names, so each individual defect type received the same total sampling weight as all "Normal" patches combined, resulting in ~92% (12/13) of training samples drawn as Defective. This caused severe training bias (train accuracy ~0.94 while validation accuracy collapsed to ~0.18-0.35).

This bug is now fixed: when `stage == "binary"`, balancing uses `record["binary_label"]` (guaranteeing ~50/50 balance between Normal and Defective). Checkpoints and results generated prior to this fix were affected by this bug and must be retrained.

---

## 8. Why U-Net

English:
AITEX provides pixel masks, not bounding boxes. Segmentation is therefore the
natural fit, and U-Net's skip connections preserve the fine spatial detail
needed to find a defect only a few pixels wide.

The architecture uses a 4-stage encoder-decoder:
- **Encoder Channels**: `64 -> 128 -> 256 -> 512`
- **Bottleneck**: `1024`
- **Decoder Channels**: `512 -> 256 -> 128 -> 64`
- **Output**: `1` (defect mask logits)
- **Parameters**: Exactly **$31,037,633$ parameters** ($\approx 31.04\text{M}$).
- **Disk Size**: At 4 bytes per float32 parameter, model weights take $31,037,633 \times 4 = 124,150,532\text{ bytes} \approx 124.2\text{MB}$. Checkpoints store weights and validation metadata only (`model_state`, `threshold`, `epoch`, `val_dice`, `val_iou`).

The shipped checkpoint (`models/segmenter/unet_best.pt`, trained Oct 8, epoch 7) was trained with **Dice + BCE loss** (`DiceBCELoss`). While Focal Tversky loss and clean-patch negative sampling (25-30% background patches) were subsequently implemented in the codebase, they have not yet been used to retrain a shipped checkpoint.

Across the 72 defective training strips with ground-truth masks, the median defect area is only **0.0087%** of the full strip (mean: 0.5550%, pulled up by a small number of large weave tears reaching up to 22.2%). At the cropped patch level (256x256), foreground defect pixels account for roughly 0.8-1.5% in typical flaw crops. Dice loss optimizes region overlap directly, preventing the model from collapsing to an all-background prediction.

This is segmentation and localization. The project does **not** do object
detection.

Hindi:
AITEX mein pixel masks hain, bounding boxes nahi. Isliye segmentation natural
fit hai, aur U-Net ke skip connections chhote defects dhoondne ke liye zaroori
hain. Shipped checkpoint (`unet_best.pt`, Oct 8) Dice + BCE loss se train hua tha.
Focal Tversky aur clean-patch sampling code mein add ho chuki hain par naya model
train nahi kiya gaya. Architecture mein 64->128->256->512->1024 channels aur
total 31.04M parameters hain jo 124.2 MB memory lete hain.

---

## 9. Why YOLO is not used yet

English:
YOLO requires bounding box annotations. AITEX does not ship them. Converting
masks into boxes is possible, but doing it would mean evaluating a detection
model on derived labels that were never human-annotated as boxes, and it would
throw away the precise boundary information the dataset actually provides.

Object detection stays a documented future experiment, not a current component.

Hindi:
YOLO ko bounding box annotations chahiye, jo AITEX mein nahi hain. Mask se box
banaya ja sakta hai, par wo derived labels honge aur asli boundary information
khoj jayegi. Isliye YOLO abhi nahi, future experiment ke roop me.

---

## 10. Train / validation / test methodology

### The mandatory rule

```
Original image split  ->  Patch generation  ->  Training
```

Never the reverse. If some patches of one original image go to train and others
to test, the model has already seen that exact fabric strip during training, and
test accuracy becomes meaningless.

Patch generation is deterministic and reads `data/split.json`, so the same
split always produces the same patches.

### Current split

| Split | Images |
| --- | --- |
| train | 173 |
| val | 37 |
| test | 37 |

`sklearn.train_test_split(stratify=...)` cannot be used. `Knots`,
`Contamination` and `Weft Crack` have one image each, and stratifying would
require one per split. `split_dataset.stratified_group_split` guarantees at
least one val and one test image for every class with three or more images, one
val image for classes with two, and keeps single-image classes in train.

### Leakage: fabric rolls

English:
There is a second, subtler leak. Normal images come from only **7 physical
fabric rolls** (`2306881-210020u`, `2306894-210033u`, `2311517-195063u`,
`2311694-1930c7u`, `2311694-2040n7u`, `2311980-185026u`, `2608691-202020u`),
each contributing up to 21 near-identical images. Defect images are grouped by
the same two-digit fabric code, so there are 8 distinct rolls in total.

A random image-level split can therefore put two near-duplicate strips from one
roll into both train and test.

`split_dataset.py` records the roll code for every image in `split.json` under
`image_rolls`, and offers `--strict-group-split`, which never splits a roll.
Verified: with strict mode, roll overlap between splits is zero.

However, **strict mode cannot be the default for this dataset**, and the numbers
show why:

| Mode | Classes in val | Classes in test | Roll overlap |
| --- | --- | --- | --- |
| `class_stratified` (default) | 10 | 9 | yes |
| `strict-group-split` | 7 | 6 | none |

With only 8 rolls, grouping by roll removes nearly half the defect classes from
validation and test. Strict mode is provided as the more conservative protocol
and is available on demand, but the default is unchanged so the current
experiment is not silently altered.

This is a genuine limitation and is stated rather than hidden.

Hindi:
Normal images sirf 7 physical fabric rolls se aati hain, aur har roll se up to
21 nearly identical images hain. Isliye ek roll ki do images train aur test
dono mein ja sakti hain. `--strict-group-split` ye rokta hai, lekin sirf 8
rolls hone ki wajah se isse zyadatar classes val/test se hat jaati hain (val 10
se 7, test 9 se 6). Isliye default nahi badla gaya.

---

## 11. Image-level inference and aggregation

English:
The classifier is trained on 256 x 256 patches, so an uploaded full strip is
**never** resized. It is padded, cut into patches, and each patch is scored.

Stage 1:
- every patch is scored independently for P(Defective)
- a patch is flagged defective when that probability exceeds
  `STAGE1_DEFECT_PROB_THRESHOLD` (0.5)
- the image is defective when at least `STAGE1_MIN_DEFECTIVE_PATCHES` (1)
  patches are flagged
- stage-1 confidence is averaged **over flagged patches only**. Averaging over
  all 16 patches would dilute one real defect to about 6% and make a confident
  detection look like noise.

Stage 2:
- runs only on flagged patches
- each flagged patch contributes a softmax vector
- vectors are averaged weighted by that patch's stage-1 defective probability
- arg-max is the image-level defect type
- reported confidence is the mean class probability across flagged patches

Taking the arg-max of the single most confident patch was rejected: a strip
holds one defect and fifteen clean patches, so a single-patch decision ignores
that majority evidence.

Localization:
- U-Net runs on the flagged patches only
- patch masks are pasted back into original image coordinates
- the padded right edge of a 3796 px image is cropped off, so the returned mask
  matches the uploaded image exactly
- a bounding box is derived from the mask as a convenience, and is explicitly
  labelled as derived from segmentation, not detection

All thresholds are configurable from `config.py` and from the Streamlit sidebar.
No inference logic lives in `app.py`; it lives in `src/inference.py` so the CLI,
evaluation and app agree.

Hindi:
Classifier 256x256 patches par train hota hai, isliye uploaded image ko resize
nahi kiya jata. Wo padded hokar patches mein kat ti hai. Stage 1 har patch ko
score karta hai, sirf flagged patches ka average confidence hota hai. Stage 2
sirf unhi patches par chalta hai aur unke probabilities weighted average hote
hain. Saara inference logic `src/inference.py` mein hai, app mein nahi.

---

## 12. Commands

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Place the dataset

```
data/raw/Defect_images/
data/raw/NODefect_images/
data/raw/Mask_images/
```

### 1. Split (must be run first)

```bash
python src/split_dataset.py
python src/split_dataset.py --strict-group-split    # optional, roll-grouped
```

### 2. Stage 1: binary classifier

```bash
python src/train_classifier.py --stage binary --model cnn
python src/train_classifier.py --stage binary --model efficientnet --finetune
```

### 3. Stage 2: defect type classifier

```bash
python src/train_classifier.py --stage defect_type --model cnn
python src/train_classifier.py --stage defect_type --model efficientnet --finetune
```

### 4. Segmentation

```bash
python src/train_segmenter.py
```

### 5. Threshold tuning (validation only, never test)

```bash
python src/tune_thresholds.py --model cnn
python src/tune_thresholds.py --model efficientnet
```

### 6. Evaluation

```bash
python src/evaluate.py --model cnn
python src/evaluate.py --model efficientnet
```

### 7. Web app

```bash
streamlit run webapp/app.py
```

### Useful flags

```bash
--epochs N              # override epoch count
--learning-rate LR      # override learning rate
--batch-size N
--patience N            # early stopping; 0 disables
--defect-threshold F    # patch defect mask fraction
--seed N
--num-workers N
--no-balance            # disable the weighted sampler (ablation)
--finetune / --no-finetune
--finetune-epochs N
```

Example, the exact comparison the project needs:

```bash
python src/train_classifier.py --stage binary --model cnn --epochs 20
python src/train_classifier.py --stage binary --model efficientnet --epochs 20 --finetune
python src/train_classifier.py --stage defect_type --model cnn --epochs 20
python src/train_classifier.py --stage defect_type --model efficientnet --epochs 20 --finetune
```

---

## 13. Training quality

- Fixed seeds for `random`, `numpy` and `torch`
- CPU/CUDA auto-detection
- `WeightedRandomSampler` for class balance (binary stage balances Normal vs Defective; defect_type balances across types)
- Cosine annealing learning-rate schedule
- Early stopping on validation balanced accuracy
- Best-checkpoint saving on validation balanced accuracy, not the final epoch
- Training and validation curves written to `results/training_curves/`
- Class distribution printed before training starts

For EfficientNet-B0, transfer learning was executed as a two-phase protocol:
1. **Phase 1 (Head training, epochs 1..17, early-stopped)**: Backbone feature extractor was completely frozen (`requires_grad=False`), backbone BatchNorm layers were locked in `eval()` mode, and only the classifier head was trained at base learning rate ($3 \times 10^{-4}$) with CosineAnnealingLR. This phase achieved its best validation balanced accuracy of **0.8901** (overall accuracy: 0.8861) at **Epoch 12**.
2. **Phase 2 (Tail fine-tuning, epochs 18..27, early-stopped)**: Deepest feature blocks (blocks $\ge 5$, comprising `features.5`, `features.6`, `features.7`, `features.8`) were unfrozen and trained with AdamW at learning rate $3 \times 10^{-5}$. Unfrozen BatchNorm layers were set to `train()` mode. Phase 2 reached its peak validation balanced accuracy of **0.8712** (overall accuracy: 0.8987) at **Epoch 24**.

**Empirical Finding**: Full Phase 2 fine-tuning did not improve upon the frozen head checkpoint (0.8712 vs 0.8901 validation balanced accuracy). Consequently, the Phase 1 checkpoint (`epoch 12`) was automatically selected by the safe-checkpoint selection logic as the winning model (`binary_efficientnet_best.pt`).

**Validation Architecture Comparison**:
- **Custom CNN (Baseline)**: Best Epoch 6, Validation Balanced Accuracy = **0.7040** (Overall Accuracy = 0.7785).
- **EfficientNet-B0 (Frozen Head)**: Best Epoch 12, Validation Balanced Accuracy = **0.8901** (Overall Accuracy = 0.8861).
- **EfficientNet-B0 (Fine-Tuned)**: Best Epoch 24, Validation Balanced Accuracy = **0.8712** (Overall Accuracy = 0.8987).

---

## 14. Evaluation metrics

`src/evaluate.py` reports three distinct levels, because they answer different
questions.

**A. Patch-level classification**
Accuracy, balanced accuracy, macro precision, macro recall, macro F1,
weighted F1, per-class precision/recall/F1, confusion matrix.

**B. Image-level classification**
The same metrics computed once per original image, which is the decision that
actually matters in inspection (defect recall, normal-image false-positive rate, balanced accuracy, accuracy, macro F1, confusion matrix). Thresholds are tuned on the validation set via `tune_thresholds.py` (test split is never used for tuning).

**C. Segmentation**
Patch level and image level: Dice, IoU, precision, recall.

Classes absent from the test split are reported as `absent_classes` and are
marked `present_in_test: false` per class. Macro averages are computed over all
12 classes, so an absent class contributes 0 rather than being quietly dropped.

### Output

```
results/
  metrics/          test_<model>.json, train_<stage>_<model>.json, stage1_thresholds_<model>.json, train_unet.json
  confusion_matrix/ binary_patch_confusion_matrix.png, defect_type_patch_confusion_matrix.png, binary_image_level_confusion_matrix.png
  training_curves/  binary_cnn_curves.png, unet_curves.png, ...
  predictions/      image-level prediction rows
  localization/     qualitative mask comparison figures
```

Note on smoke-test directories: `models/_smoketest_archive`, `models/_smoketest_archive_v1`, `results/_smoketest_archive`, and `results/_smoketest_archive_v1` are 1-2 epoch smoke tests from early development and are not results.

---

## 15. Streamlit usage

```bash
streamlit run webapp/app.py
```

The UI shows:

1. uploaded image
2. Normal or Defective, with defect type and confidence
3. predicted mask
4. highlighted overlay
5. derived bounding box
6. optional patch-level table

Sidebar controls: model choice, stage-1 probability threshold (auto-loaded from tuned validation thresholds if available), minimum
defective patches, U-Net mask threshold, patch table toggle.

`app.py` contains UI only. Missing or mismatched checkpoints produce a clear
error naming the exact training command to run.

---

## 16. Limitations

Honest assessment:

1. **Small Test Set and Wide Confidence Intervals**: The held-out test split contains only 37 images (21 Normal, 16 Defective). Stage 1 test balanced accuracy is 0.9137 with approximate 95% confidence interval `[0.7066, 0.9783]` (averaging the Wilson interval endpoints of Recall and TNR). Individual Wilson score intervals remain wide: defect recall 0.8750 `[0.6398, 0.9650]` and normal FPR 0.0476 `[0.0085, 0.2267]`.
2. **Borderline $k=2$ Decision Rule**: Defects concentrated in a single patch are missed by the minimum patch rule (two test false negatives: `0001_002_00.png` and `0058_019_06.png`, each with exactly 1 patch $> 0.60$). Conversely, normal fabric texture can trigger multiple patches (one false positive: `0013_000_03.png`, with 9 patches $> 0.60$ under the $\tau = 0.60, k = 2$ decision threshold).
3. **Stage 2 Defect Type Classification is Weak**: Test accuracy is 0.4118 (14/34) and the model frequently defaults to *Broken Pick*. Ten classes have $\le 2$ test samples, and three classes (*Knots*, *Contamination*, *Weft Crack*) have only one original image and remain train-only.
4. **Preliminary Localization**: The U-Net achieves a mean Dice of 0.1575, but the median Dice is only 0.0814. It touches the ground-truth box in 8 of 14 defective test images ($57.1\%$).
5. **Roll Leakage in Default Split**: Fabric weave patterns come from only 8 distinct rolls; strict roll-grouping eliminates roll leakage but removes nearly half the defect classes from validation and test due to data scarcity.
6. **No Certified Industrial Quality**: Lighting, camera resolution, loom speed, and fabric textures in actual textile mills differ substantially from static benchmark captures.

---

## 17. Future improvements

- Grouped fabric-roll evaluation as the primary protocol once more fabric is
  available
- Fabric-roll-level augmentation, or synthetic defect generation
- More defect classes with enough images to support a real test split
- Per-class threshold tuning for the binary stage
- Attention or transformer segmentation backbones
- Bounding-box object detection derived from masks, as a documented experiment
  compared against U-Net
- Quantisation and ONNX export for deployment
- A real-line pilot with a calibrated acceptance threshold

---

## 18. Results

*All numbers below are generated directly from the JSON files in `results/metrics/` using `python scripts/make_results_table.py`.*

### Stage 1 (Binary Classification) Image-Level Performance (Side-by-Side)

| Metric | Validation Set (37 images) | Test Set (37 images) | Test 95% Wilson CI |
| :--- | :---: | :---: | :---: |
| **Decision Thresholds ($\tau, k$)** | $\tau = 0.60, k = 2$ | $\tau = 0.60, k = 2$ | (Tuned on Val) |
| **Confusion Matrix [TN, FP / FN, TP]** | `[[18, 3], [2, 14]]` | `[[20, 1], [2, 14]]` | — |
| **Defect Recall (Sensitivity)** | 0.8750 (14/16) | 0.8750 (14/16) | `[0.6398, 0.9650]` |
| **Normal False Positive Rate (FPR)** | 0.1429 (3/21) | 0.0476 (1/21) | `[0.0085, 0.2267]` |
| **Balanced Accuracy** | 0.8661 | 0.9137 | `[0.7066, 0.9783]`* |
| **Overall Accuracy** | — | 0.9189 (34/37) | — |
| **Macro F1 Score** | — | 0.9167 | — |

\* *Note: The balanced accuracy 95% interval averages the Wilson interval endpoints of Recall and TNR: `[(Recall_low + TNR_low)/2, (Recall_high + TNR_high)/2]`.*

### Stage 1 Classifier Validation Ablation (Validation Set Only)

| Architecture / Phase | Best Epoch | Phase | Val Balanced Acc | Val Overall Acc | Unfrozen Blocks |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Custom CNN (Baseline)** | 6 | head | 0.7040 | 0.7785 | All layers trained from scratch |
| **EfficientNet-B0 (Frozen Head)** | 12 | head | 0.8901 | 0.8861 | Classifier head only (backbone frozen) |
| **EfficientNet-B0 (Fine-Tuned)** | 24 | fine_tune | 0.8712 | 0.8987 | Head + `features.5..8` unfrozen |

*Finding: Fine-tuning phase 2 was run across blocks `features.5..8`, but reached validation balanced accuracy 0.8712 (epoch 24), which did not improve upon the frozen-head checkpoint (0.8901, epoch 12). Therefore, the frozen head checkpoint was selected as `binary_efficientnet_best.pt`.*

### Defect Localization Performance (U-Net on 14 Scored Defective Test Images)

| Localization Metric | Mean Value | Median Value | Min | Max | Benchmark Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dice Coefficient** | 0.1575 | 0.0814 | 0.0001 | 0.6755 | One sample (`0010_006_02.png`) reaches Dice 0.6755 |
| **Intersection over Union (IoU)** | 0.1046 | 0.0568 | 0.0001 | 0.5101 | Preliminary pixel overlap |
| **Bounding Box Touch Rate** | 0.5714 | — | — | — | Touches ground-truth box in **8 of 14** images |

### Test-set Statement
"The test split was scored four or more times: for the original model, and three times for the retrained model (two invocations crashed in the reporting code after inference and were rerun after bug fixes, with no change to model, thresholds or data). Model selection, early stopping and thresholds used the validation split only. The author had seen the original model's test errors before retraining."

### Honest Assessment of Mistakes and Edge Cases
1. **Checkpoint Overwrite Incident**: The original EfficientNet checkpoint was overwritten by a short test run and subsequently retrained from scratch on GPU (checkpoint overwrite protection and timestamped backups were implemented in `src/training.py` to prevent this from recurring). The old and new results correspond to different trained models.
2. **Stage 1 Decision Borderlines**: On the test set, Stage 1 misclassifies exactly 3 images:
   - Two false negatives: `0001_002_00.png` (true: *Broken End*) and `0058_019_06.png` (true: *Fuzzyball*). In both strips, exactly 1 patch exceeds the probability threshold $\tau = 0.60$ ($P=0.9729$ and $P=0.9424$ respectively), but because $k=2$ patches are required, the strip is declared Normal.
   - One false positive: `0013_000_03.png` (true: *Normal*), where 9 of 16 patches exceeded the threshold ($P_{\max} = 0.7860$).
   - These same three images were errors in the old model run. The $k=2$ rule is borderline for several strips.
3. **Stage 2 Defect Type Classification is Weak**:
   - Defect type accuracy on the test set is 0.4118 (14/34 correct). In an earlier run, it reported 0.4706 because two near-tied borderline patches (`0075_010_03.png` patches #25 and #26) had probability margins under 0.002 between *Broken Pick* and *Warp Ball* and flipped across GPU/CPU floating-point implementations.
   - The model frequently defaults towards *Broken Pick* (the majority flaw class).
   - Ten defect classes have insufficient test data ($\le 2$ samples), and three classes (*Knots*, *Contamination*, *Weft Crack*) have only a single image in the entire dataset, so they appear only in training and cannot be evaluated on test data.
4. **Localization is Preliminary**:
   - U-Net pixel-level localization achieves a mean Dice of 0.1575, but the median Dice is 0.0814.
   - The predicted bounding box touches the ground-truth flaw box in 8 of 14 images ($57.1\%$).
   - One favorable image (`0010_006_02.png`) reaches Dice 0.6755, but most fine yarn defects have minimal pixel overlap. U-Net localization should be considered a preliminary guide for visual inspection rather than a certified segmentation tool.

---

## 19. How to Run

### Installation
```bash
git clone <YOUR-REPO-URL>
cd fabric-defect-detection
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```
*(Note: Trained `.pt` checkpoint files in `models/` are large binaries and are not tracked in git. Train or place checkpoints prior to running).*

### Web Application
```bash
streamlit run webapp/app.py
```
Open `http://localhost:8501` in your browser. Use the sidebar to switch between Balanced, Sensitive, and Strict decision styles, upload full-strip images, or select demo samples from the Sample Gallery.

### Model Evaluation
```bash
python src/evaluate.py --model efficientnet --stage all
```

### Reproducing Results Tables
```bash
python scripts/make_results_table.py
```

---

## Project structure

```
fabric-defect-detection/
├── data/
│   ├── raw/{Defect_images,NODefect_images,Mask_images}/
│   └── split.json                 # generated
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_custom_cnn.ipynb
│   ├── 04_efficientnet.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_localization.ipynb
├── src/
│   ├── config.py                  # all settings
│   ├── labels.py                  # codes and label spaces
│   ├── utils.py                   # image/mask/patch helpers
│   ├── split_dataset.py
│   ├── dataset.py                 # patch datasets
│   ├── models.py                  # CustomCNN, EfficientNet-B0, U-Net, losses
│   ├── training.py                # classifier training loop
│   ├── segmentation_training.py   # U-Net training loop
│   ├── train_classifier.py        # stage 1 + stage 2 CLI
│   ├── train_segmenter.py
│   ├── tune_thresholds.py         # validation threshold grid-search
│   ├── inference.py               # two-stage pipeline + aggregation
│   ├── predict.py                 # backward-compat shim
│   └── evaluate.py
├── models/{classifier,segmenter}/
├── results/{metrics,confusion_matrix,training_curves,predictions,localization}/
├── webapp/app.py
├── requirements.txt
└── README.md
```

---

## Disclaimer

This project is an academic research prototype. It must not be treated as a
certified industrial quality-control system without substantial further
validation on production fabric.
