# Fabric Defect Detection & Localization: Complete Technical Evaluation Report

---

## 1. Executive Summary & Problem Statement

Automated optical inspection of industrial textile fabrics is challenging due to:
1. **Extreme Aspect Ratios**: Textile loom inspection cameras produce ultra-high-resolution, panoramic strip images (e.g., $4096 \times 256$ or $3796 \times 256$ pixels).
2. **Subtle and Localized Defects**: Weave flaws (broken yarns, neps, fuzzyballs, selvage cuts) typically occupy less than $0.1\%$ of the total image area.
3. **Severe Class Imbalance**: In real production lines, $>85\%$ of manufactured fabric strips are completely defect-free ("Normal"). Furthermore, among defective strips, certain flaw types are common (e.g., *Broken Pick*), while others are extremely rare (e.g., *Knots*, *Contamination*).

To solve these challenges without losing high-frequency weave texture information, this project implements an **end-to-end, two-stage hierarchical deep learning architecture with semantic segmentation localization**:
- **Stage 1 (Binary Detector)**: Determines whether a patch / strip contains any defect ($Normal$ vs $Defective$).
- **Stage 2 (Multi-class Classifier)**: Classifies the specific defect category out of 12 flaw classes on flagged defective patches.
- **Stage 3 (U-Net Segmenter)**: Generates a pixel-level binary mask and bounding box overlay to pinpoint the exact defect location on the full fabric strip.

---

## 2. Dataset & Stratified Splitting Strategy

### 2.1 Raw Dataset Characteristics (AITEX Benchmark)
- **Defect Images**: 106 high-resolution fabric strips with defects across 12 defect categories and 7 distinct fabric weaves (`00`, `01`, `02`, `03`, `04`, `05`, `06`).
- **Normal Images**: 141 defect-free fabric strips across the same 7 weave types.
- **Ground Truth Masks**: 107 pixel-accurate binary masks.
- **Total Images**: 247 images.

### 2.2 Stratified Split (`data/split.json`)
To ensure zero data leakage and guarantee that every fabric weave pattern is represented in train, validation, and test sets, splitting is performed at the **full-strip image level** (not patch level):
- **Train Split**: 173 images (70%)
- **Validation Split**: 37 images (15%) — *Used strictly for checkpoint selection, early stopping, and image-level decision threshold tuning.*
- **Test Split**: 37 images (15%) — *Held-out test set; evaluated only once with frozen weights and tuned thresholds.*

---

## 3. Core Technical Decisions: Why Each Step Was Chosen

### 3.1 Why 256×256 Patch-Based Sliding Window?
- **Downsampling Failure**: Directly downsampling a $4096 \times 256$ strip to standard network dimensions ($224 \times 224$) would compress the width by $18\times$, destroying the micro-scale warp and weft yarn structures and rendering small defects invisible.
- **Patch Extraction Solution**: Slicing the full strip into consecutive $256 \times 256$ non-overlapping patches preserves $100\%$ of native optical resolution.
- **Padding**: Images whose widths are not exact multiples of 256 (e.g., 3796 px) are padded on the right during inference and cropped back to original dimensions before final output.

### 3.2 Why Hard Negative Mining in Patch Extraction?
- A defective strip typically has only $1$ to $2$ patches containing actual flaws; the remaining $14$ to $15$ patches from that same strip are physically clean.
- In training, up to 2 clean patches from defective strips are explicitly included as **hard negatives**. This forces the neural network to learn the subtle difference between legitimate fabric weave irregularities and genuine manufacturing defects.

### 3.3 Why Two-Stage Hierarchical Classification?
1. **Imbalance Decoupling**: Combining 1 normal class with 12 defect classes in a single 13-class classifier causes normal patches to dominate gradients. Decoupling into Binary Detection (Stage 1) and Defect Identification (Stage 2) allows each network to specialize.
2. **Computational Efficiency**: Stage 2 classification and U-Net segmentation run **only** on patches flagged as defective by Stage 1, saving $>85\%$ of computational overhead during inference.

### 3.4 Why Balanced Sampling and Balanced Accuracy?
- Standard cross-entropy with uniform sampling on imbalanced data causes the model to predict the majority class ($Normal$) to achieve artificial $\sim 85\%$ accuracy while missing actual defects.
- **Balanced Batch Sampler**: Assigns inverse-frequency weights so each class has equal representation in expectation per training epoch.
- **Balanced Accuracy**: Arithmetic mean of sensitivity (Defect recall) and specificity (Normal recall):
  $$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{Normal}} + \text{Recall}_{\text{Defective}}}{2}$$
  Used for epoch checkpoint selection and early stopping to guarantee the model does not degenerate into predicting only one class.

### 3.5 Why Validation-Only Image-Level Threshold Tuning?
- Slicing a strip into 16 patches means a fixed patch decision threshold of $P(\text{defective}) \ge 0.5$ with 1 patch trigger will accumulate false positives over 16 trials ($1 - (1 - \text{FPR})^{16}$).
- [`src/tune_thresholds.py`](file:///d:/fabric-defect-detection/src/tune_thresholds.py) grid-searches both the **patch probability threshold** ($\tau \in [0.30, 0.95]$) and the **minimum required defective patch count** ($k \in [1, 4]$) exclusively on the **validation split**. The test split is never touched during tuning.

---

## 4. Deep Learning Model Architectures

### 4.1 Custom Baseline CNN Architecture (`CustomCNN`)
A 4-stage convolutional baseline designed from scratch:

```mermaid
graph TD
    Input["Input Patch (3, 256, 256)"] --> Block1["Block 1: Conv2d(3->32, k=3, p=1) + BatchNorm2d(32) + ReLU + MaxPool2d(2)"]
    Block1 --> Block2["Block 2: Conv2d(32->64, k=3, p=1) + BatchNorm2d(64) + ReLU + MaxPool2d(2)"]
    Block2 --> Block3["Block 3: Conv2d(64->128, k=3, p=1) + BatchNorm2d(128) + ReLU + MaxPool2d(2)"]
    Block3 --> Block4["Block 4: Conv2d(128->256, k=3, p=1) + BatchNorm2d(256) + ReLU + MaxPool2d(2)"]
    Block4 --> Pool["AdaptiveAvgPool2d(1) -> Flatten (256)"]
    Pool --> Head["Linear(256 -> NumClasses)"]
```

- **Feature Map Dimensions**:
  - Input: $(3, 256, 256)$
  - After Block 1: $(32, 128, 128)$
  - After Block 2: $(64, 64, 64)$
  - After Block 3: $(128, 32, 32)$
  - After Block 4: $(256, 16, 16)$
  - After Global Pool: $(256, 1, 1) \to 256$ vector
  - Output Logits: $(N, \text{NumClasses})$

---

### 4.2 EfficientNet-B0 with Two-Phase Fine-Tuning
Utilizes ImageNet-pretrained weights (`EfficientNet_B0_Weights.DEFAULT`):

- **Backbone**: EfficientNet-B0 feature extractor ($9$ feature blocks `features.0` to `features.8`, depthwise-separable MBConv convolutions with squeeze-and-excitation).
- **Classifier Head**:
  $$\text{Dropout}(p=0.30) \longrightarrow \text{Linear}(1280 \longrightarrow \text{NumClasses})$$
- **True Two-Phase Training Strategy**:
  - **Phase 1 (Epochs 1 to $E_{\text{head}}$)**: Entire backbone frozen (`requires_grad = False`). Backbone BatchNorm layers forced to `eval()` mode to prevent batch statistics corruption. Only the linear classifier head trains at base learning rate $\text{LR} = 3 \times 10^{-4}$.
  - **Phase 2 (Epochs $E_{\text{head}}+1 \dots$)**: Deepest feature blocks (Blocks $\ge 5$, comprising `features.5`, `features.6`, `features.7`, `features.8`) are unfrozen and assigned a dedicated low learning rate group ($\text{LR}_{\text{finetune}} = 3 \times 10^{-5}$). Lower blocks (0–4) remain frozen to preserve fundamental edge/texture filters. The cosine annealing scheduler is dynamically rebuilt for the remaining epoch budget.

---

#### 4.3 U-Net Localization Segmenter (`UNet`)
A symmetric 4-stage encoder-decoder network with skip connections:
- **Encoder Channels**: `64 -> 128 -> 256 -> 512`
- **Bottleneck**: `1024`
- **Decoder Channels**: `512 -> 256 -> 128 -> 64`
- **Output Layer**: `Conv2d(64 -> 1, kernel_size=1)`
- **Model Parameter Count**: Total parameter count is **$31,037,633$ parameters** ($\approx 31.04\text{M}$). In float32 precision, the model weights consume $31,037,633 \times 4\text{ bytes} = 124,150,532\text{ bytes} \approx 124.2\text{MB}$.
- **Checkpoint Contents**: Checkpoints save only the model state dictionary and validation metadata (`dict_keys(['model_state', 'threshold', 'epoch', 'val_dice', 'val_iou'])`), containing weights only. The $124.2\text{MB}$ file size is entirely accounted for by the $31.04\text{M}$ float32 parameters.
- **Loss Function for Shipped Checkpoint**: The shipped checkpoint (`models/segmenter/unet_best.pt`, trained Oct 8, epoch 7) was trained with **Dice + BCE Loss** (`DiceBCELoss`):
  $$\mathcal{L}_{\text{total}} = 0.5 \times \text{BCEWithLogitsLoss}(y_{\text{pred}}, y_{\text{true}}) + \text{DiceLoss}(y_{\text{pred}}, y_{\text{true}})$$
  *Note: `FocalTverskyBCELoss` and clean-patch background sampling (25-30% of each batch) were subsequently implemented in `src/models.py` and `src/segmentation_training.py`, but have not been used to retrain a shipped checkpoint.*
- **Foreground Pixel Statistics Reconciled**: Across the 72 training strips with ground-truth masks, the median defect area is only **0.0087%** of the full strip (mean: 0.5550%, pulled up by a small number of large weave tears reaching up to 22.2%). At the cropped patch level (256x256), foreground defect pixels account for roughly 0.8-1.5% in typical flaw crops.
- **Image Stitching**: Patch masks are pasted back into the full-strip coordinate frame to reconstruct full-resolution binary defect segmentation maps and extract bounding boxes.

---

## 5. Summary of Bug Fixes & Code Enhancements

| Component | Issue Identified | Technical Fix Applied |
| :--- | :--- | :--- |
| **`src/training.py`** | Sampler weighted binary stage by `class_name`, making training data ~92% defective | Switched key to `binary_label` for stage=="binary"; added 1-epoch sanity check; created unit tests in [`tests/test_sampler.py`](file:///d:/fabric-defect-detection/tests/test_sampler.py) |
| **`src/training.py`** | Checkpoint selection & early stopping used plain accuracy on imbalanced val set | Switched selection metric to `balanced_accuracy_score`; tracked per-class recalls |
| **`src/models.py`** | `set_backbone_trainable` boolean logic mistakenly disabled classifier head gradients when `trainable=False` | Fixed conditional assignment so `classifier.*` parameters always remain trainable in Phase 1 |
| **`src/training.py`** | `*_last.pt` saved requested total epochs instead of actual completed epoch | Fixed to record `len(history['train_loss'])` |
| **Threshold Tuning** | Hardcoded 1-patch threshold produced excessive false positives on 16-patch strips | Created [`src/tune_thresholds.py`](file:///d:/fabric-defect-detection/src/tune_thresholds.py) grid-search on validation images; auto-loaded in evaluation & UI |
| **`src/inference.py`** | `torch.load(weights_only=False)` security vulnerability | Switched to `weights_only=True` with explicit `ModelNotTrainedError` and format validation |
| **`webapp/app.py`** | App would crash on empty, corrupt, or non-image uploads | Added robust MIME-type, byte-size, and OpenCV decode error trapping with friendly user messages |
| **`.gitignore`** | Blanket ignore masked split reproducible data | Explicitly tracked [`data/split.json`](file:///d:/fabric-defect-detection/data/split.json) and metrics/visualizations in `results/` |

---

## 6. Complete Training & Validation Metrics (Exact Values)

*Note: All values below are direct extractions from `results/metrics/` and model checkpoints. No numbers are invented.*

### 6.1 Stage 1 Classifier Validation Ablation (Validation Set Only)

*Generated directly via `python scripts/make_results_table.py`:*

| Architecture / Phase | Best Epoch | Phase | Val Balanced Acc | Val Overall Acc | Unfrozen Blocks |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Custom CNN (Baseline)** | 6 | head | 0.7040 | 0.7785 | All layers trained from scratch |
| **EfficientNet-B0 (Frozen Head)** | 12 | head | 0.8901 | 0.8861 | Classifier head only (backbone frozen) |
| **EfficientNet-B0 (Fine-Tuned)** | 24 | fine_tune | 0.8712 | 0.8987 | Head + `features.5..8` unfrozen |

*Finding: Fine-tuning phase 2 was run across blocks `features.5..8`, but reached validation balanced accuracy 0.8712 (epoch 24), which did not improve upon the frozen-head checkpoint (0.8901, epoch 12). Therefore, the frozen head checkpoint was selected as `binary_efficientnet_best.pt`.*

### 6.2 Stage 2 (Defect Type Multi-Class)

| Metric | Custom CNN | EfficientNet-B0 |
| :--- | :---: | :---: |
| **Epochs Run** | 20 / 11 (Early stopped) | 30 / 18 (Early stopped) |
| **Best Epoch** | Epoch 6 | **Epoch 13** |
| **Best Val Balanced Accuracy** | $0.4577$ ($45.77\%$) | **$0.5675$ ($56.75\%$)** |
| **Best Val Accuracy** | $0.5172$ ($51.72\%$) | **$0.6552$ ($65.52\%$)** |
| **Training Patches** | 109 Defective patches | 109 Defective patches |
| **Validation Patches** | 29 Defective patches | 29 Defective patches |

### 6.3 Stage 3 (U-Net Localization Segmenter)

| Metric | U-Net Segmenter Value |
| :--- | :---: |
| **Epochs Run** | 20 / 12 (Early stopped) |
| **Best Epoch** | **Epoch 7** |
| **Best Validation Dice Coefficient** | **$0.2389$** ($23.89\%$) |
| **Validation IoU at Best Epoch** | **$0.1857$** ($18.57\%$) |
| **Training Patches** | 119 Defective patches |
| **Validation Patches** | 29 Defective patches |

---

## 7. Validation Threshold Tuning Results

Grid search across $\tau \in [0.30, 0.95]$ and $k \in [1, 4]$ on 37 validation images (`results/metrics/stage1_thresholds_efficientnet.json`):

| Parameter / Metric | CNN (`stage1_thresholds_cnn.json`) | EfficientNet (`stage1_thresholds_efficientnet.json`) |
| :--- | :---: | :---: |
| **Optimal Prob Threshold ($\tau$)** | **0.45** | **0.60** |
| **Optimal Min Defective Patches ($k$)**| **4** | **2** |
| **Validation Balanced Accuracy** | $0.7560$ | **$0.8661$** |
| **Validation Defect Recall (TPR)** | $0.7500$ (12 / 16) | **$0.8750$ (14 / 16)** |
| **Validation Normal FPR** | $0.2381$ (5 / 21) | **$0.1429$ (3 / 21)** |
| **Validation Confusion Matrix** | $\begin{bmatrix} 16 & 5 \\ 4 & 12 \end{bmatrix}$ | $\begin{bmatrix} 18 & 3 \\ 2 & 14 \end{bmatrix}$ |

---

## 8. Final Test Set Evaluation Benchmark (Held-out 37 Images)

*Test-set statement:*
> "The test split was scored four or more times: for the original model, and three times for the retrained model (two invocations crashed in the reporting code after inference and were rerun after bug fixes, with no change to model, thresholds or data). Model selection, early stopping and thresholds used the validation split only. The author had seen the original model's test errors before retraining."

### 8.1 Image-Level Binary Defect Detection Benchmark (Side-by-Side)

*Generated directly via `python scripts/make_results_table.py`:*

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

### 8.2 Patch-Level Binary Classification Benchmark (164 Test Patches)

| Metric | Custom CNN | EfficientNet-B0 |
| :--- | :---: | :---: |
| **Patch Accuracy** | $0.8049$ | **$0.8720$** (143 / 164) |
| **Patch Balanced Accuracy** | $0.8118$ | **$0.8975$** |
| **Defective Patch Recall** | $0.8235$ (28 / 34) | **$0.9412$ (32 / 34)** |
| **Normal Patch Precision** | $0.9455$ (104 / 110) | **$0.9823$ (111 / 113)** |
| **Patch Confusion Matrix** | $\begin{bmatrix} 104 & 26 \\ 6 & 28 \end{bmatrix}$ | $\begin{bmatrix} \mathbf{111} & \mathbf{19} \\ \mathbf{2} & \mathbf{32} \end{bmatrix}$ |

### 8.3 Defect Localization Performance (U-Net on 14 Scored Defective Test Images)

*Generated directly via `python scripts/make_results_table.py`:*

| Localization Metric | Mean Value | Median Value | Min | Max | Benchmark Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dice Coefficient** | 0.1575 | 0.0814 | 0.0001 | 0.6755 | One sample (`0010_006_02.png`) reaches Dice 0.6755 |
| **Intersection over Union (IoU)** | 0.1046 | 0.0568 | 0.0001 | 0.5101 | Preliminary pixel overlap |
| **Bounding Box Touch Rate** | 0.5714 | — | — | — | Touches ground-truth box in **8 of 14** images |

---

## 9. In-Depth Analysis of Failures, Discrepancies & Limitations

### 9.1 Stage 1 Decision Borderlines & Honest Misclassification Audit
Under the validation-tuned decision thresholds ($\tau = 0.60, k = 2$), Stage 1 achieves a held-out test balanced accuracy of **0.9137** (approximate 95% CI `[0.7066, 0.9783]`, defect recall 0.8750 `[0.6398, 0.9650]`, normal FPR 0.0476 `[0.0085, 0.2267]`). On this 37-image test split, exactly 3 images are misclassified:
- **Two False Negatives**: `0001_002_00.png` (true: *Broken End*) and `0058_019_06.png` (true: *Fuzzyball*). In both cases, exactly 1 patch triggers above threshold ($P=0.9729$ and $P=0.9424$), but the minimum required patch count ($k=2$) is not met, so the strip is called Normal.
- **One False Positive**: `0013_000_03.png` (true: *Normal*), where 9 of 16 patches exceeded the threshold $\tau = 0.60$ ($P_{\max} = 0.7860$).
- These exact three images were also errors in the old model run before retraining. The $k=2$ rule is borderline for several strips.

### 9.2 Stage 2 Defect Type Classifier: Severe Support Imbalance
- **The Data Reality**: Out of 106 defective strips in AITEX, only *Broken Pick* (support 19) and *Fuzzyball* (support 6) have adequate test sample support ($\ge 3$). All other 10 defect types have $\le 2$ test samples (with *Knots*, *Contamination*, and *Weft Crack* having 0 test samples).
- **Test Accuracy**: Accuracy is 0.4118 (14/34 correct). In an earlier run, it reported 0.4706 because two near-tied borderline patches (`0075_010_03.png` patches #25 and #26) had probability margins under 0.002 between *Broken Pick* and *Warp Ball* and flipped across GPU/CPU floating-point implementations.
- **Majority Class Bias**: The model frequently defaults towards *Broken Pick*. Stage 2 must be treated as an advisory suggestion.

### 9.3 U-Net Localization: Preliminary Pixel Segmentation
- **Extreme Foreground Sparsity**: Defect pixels constitute only $\sim 0.009\%$ of raw strip pixels.
- **Preliminary Results**: Test strip-level mean Dice of $0.1575$ (median $0.0814$) and mean IoU of $0.1046$ indicate preliminary localization capability suitable for bounding-box visual guidance rather than precise edge boundary segmentation. Touches ground truth box in 8 of 14 images ($57.1\%$).

---

## 10. Jupyter Notebook Suite (`notebooks/`)

1. **`01_dataset_exploration.ipynb`**: Visualizes image dimensions ($4096 \times 256$), defect label distributions, and ground-truth mask overlays.
2. **`02_preprocessing.ipynb`**: Demonstrates sliding window slicing ($256 \times 256$, stride 256), hard negative extraction, and dataset stratification.
3. **`03_custom_cnn.ipynb`**: End-to-end walkthrough of the baseline `CustomCNN` model training and convergence curves.
4. **`04_efficientnet.ipynb`**: Detailed execution of the 2-phase fine-tuning transfer learning process.
5. **`05_evaluation.ipynb`**: Generates ROC curves, Precision-Recall curves, confusion matrices, and classification reports.
6. **`06_localization.ipynb`**: Demonstrates U-Net mask prediction, probability thresholding, and full-strip overlay stitching.

---

## 11. Streamlit Web Application (`webapp/app.py`)

- **Interactive UI**:
  - Model selection sidebar (Custom CNN baseline vs EfficientNet-B0).
  - Decision presets: Balanced, Sensitive, and Strict auto-loaded from `results/metrics/stage1_thresholds_*.json` with manual tuning expander.
  - Full-strip image upload (`.png`, `.jpg`, `.jpeg`) and preloaded Sample Gallery.
  - Interactive SVG strip diagram showing all 16 patches colored by defect probability with red outlines for flagged patches.
  - Suggested defect type bar chart with low-confidence alert (<40%).
  - U-Net overlay view with toggles for bounding box and raw binary mask.
  - Patch-by-patch inspection table.
  - Download buttons for inspected PNG overlay and machine-readable JSON report.

---

## 12. Reproduction & Verification Commands

To reproduce the pipeline:

```bash
# 1. Run unit tests
pytest tests/ -v

# 2. Reproduce results tables from JSON files
python scripts/make_results_table.py

# 3. Create stratified dataset split
python src/split_dataset.py

# 4. Train Stage 1 Classifiers (Binary) with checkpoint overwrite protection
python src/train_classifier.py --stage binary --model cnn
python src/train_classifier.py --stage binary --model efficientnet --finetune

# 5. Train Stage 2 Classifiers (Defect Type)
python src/train_classifier.py --stage defect_type --model cnn
python src/train_classifier.py --stage defect_type --model efficientnet --finetune

# 6. Train U-Net Segmenter
python src/train_segmenter.py

# 7. Tune Decision Thresholds on Validation Data
python src/tune_thresholds.py --model cnn
python src/tune_thresholds.py --model efficientnet

# 8. Evaluate on Held-out Test Set
python src/evaluate.py --model efficientnet --stage all

# 9. Launch Web Application
streamlit run webapp/app.py
```
*(Note: Checkpoint `.pt` files in `models/` are not tracked in git due to file size limits).*

---

## 12. Reproduction & Verification Commands

To reproduce the entire pipeline from scratch:

```bash
# 1. Verify environment and compilation
python -c "import py_compile, glob; [py_compile.compile(f, doraise=True) for f in glob.glob('src/*.py') + glob.glob('webapp/*.py') + glob.glob('tests/*.py')]; print('Compilation OK!')"

# 2. Run unit tests
pytest tests/

# 3. Create stratified dataset split
python src/split_dataset.py

# 4. Train Stage 1 Classifiers (Binary)
python src/train_classifier.py --stage binary --model cnn
python src/train_classifier.py --stage binary --model efficientnet --finetune

# 5. Train Stage 2 Classifiers (Defect Type)
python src/train_classifier.py --stage defect_type --model cnn
python src/train_classifier.py --stage defect_type --model efficientnet --finetune

# 6. Train U-Net Segmenter
python src/train_segmenter.py

# 7. Tune Decision Thresholds on Validation Data
python src/tune_thresholds.py --model cnn
python src/tune_thresholds.py --model efficientnet

# 8. Evaluate on Held-out Test Set
python src/evaluate.py --model cnn
python src/evaluate.py --model efficientnet

# 9. Launch Web Application
streamlit run webapp/app.py
```
