"""English: Deep-learning model definitions.
Hindi: Project ke neural-network architectures ko ek separate module mein rakha gaya hai
taaki training aur inference dono same model definitions use karein.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class CustomCNN(nn.Module):
    """English:
    Small CNN baseline.
    Hindi:
    Ye hamara simple baseline model hoga. Isse hum samajh sakte hain ki
    pretrained model se kitna improvement mil raha hai.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_efficientnet(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """English:
    EfficientNet-B0 with transfer learning.
    Hindi:
    Ye ImageNet pretrained model hai. Ise fabric defect classification ke liye
    adapt kiya gaya hai.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """English: Freeze or unfreeze every parameter except the classifier head.
    Hindi: Classifier head ke alawa saare parameters ko freeze/unfreeze karta hai.
    """
    for name, parameter in model.named_parameters():
        if name.startswith("classifier"):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = trainable


def unfreeze_last_blocks(model: nn.Module, from_block: int) -> List[str]:
    """English: Unfreeze the tail of the EfficientNet feature extractor.
    Hindi: EfficientNet feature extractor ke aakhri blocks ko unfreeze karta hai.

    English: EfficientNet-B0 has nine ``features`` blocks (0-8). Fine-tuning
    only the deepest blocks keeps the low-level ImageNet features intact,
    which matters for fabric because weave texture is a low-level cue.
    Hindi: EfficientNet-B0 mein nau ``features`` blocks hote hain (0-8).
    Sirf gehre blocks fine-tune karne se low-level ImageNet features bache
    rehte hain, jo fabric ke liye zaroori hain.
    """
    unfrozen: List[str] = []
    blocks = list(model.features.children())
    for position, block in enumerate(blocks):
        trainable = position >= from_block
        for parameter in block.parameters():
            parameter.requires_grad = trainable
        if trainable:
            unfrozen.append(f"features.{position}")
    return unfrozen


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """English: Factory used by training and inference.
    Hindi: Training aur inference dono ke liye model factory.
    """
    if model_name == "cnn":
        return CustomCNN(num_classes)
    if model_name == "efficientnet":
        return build_efficientnet(num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unknown model '{model_name}'. Use 'cnn' or 'efficientnet'."
    )


class DoubleConv(nn.Module):
    """English: Convolution + BatchNorm + ReLU applied twice.
    Hindi: Do baar convolution-normalisation-ReLU lagata hai. Ye U-Net ka
    basic building block hai.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """English:
    U-Net for defect mask prediction.
    Hindi:
    U-Net architecture. Ye encoder-decoder structure ke saath skip connections
    use karta hai, jisse chhote defects bhi accurately locate ho paate hain.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)
        self.down3 = self._down(256, 512)
        self.down4 = self._down(512, 1024)

        self.up1 = self._up(1024, 512)
        self.up2 = self._up(512, 256)
        self.up3 = self._up(256, 128)
        self.up4 = self._up(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    @staticmethod
    def _down(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    @staticmethod
    def _up(in_channels: int, out_channels: int) -> nn.Module:
        return nn.ModuleList(
            [
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                DoubleConv(in_channels // 2 + out_channels, out_channels),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        up1 = self.up1[0](x5)
        up1 = self.up1[1](torch.cat([up1, x4], dim=1))
        up2 = self.up2[0](up1)
        up2 = self.up2[1](torch.cat([up2, x3], dim=1))
        up3 = self.up3[0](up2)
        up3 = self.up3[1](torch.cat([up3, x2], dim=1))
        up4 = self.up4[0](up3)
        up4 = self.up4[1](torch.cat([up4, x1], dim=1))

        return self.outc(up4)


def dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """English: Dice loss computed safely on probabilities.
    Hindi: Dice loss probabilities par safely compute hota hai.

    English: ``smooth`` is added to both numerator and denominator so a patch
    with no predicted and no true defect pixels yields a loss of exactly 1.0
    instead of dividing by zero. This matters because the AITEX foreground is
    roughly 0.9% of pixels, so empty-prediction batches are common.
    Hindi: ``smooth`` numerator aur denominator dono mein add hota hai taaki
    koi bhi patch zero prediction par division-by-zero na de.
    """
    intersection = (probs * targets).sum(dim=(2, 3))
    cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """English:
    Combination of Dice loss and binary cross-entropy.
    Hindi:
    Dice loss region overlap ko optimise karta hai aur binary cross-entropy
    pixel-wise accuracy improve karta hai. AITEX mein defect area bahut chhota
    hota hai (median ~0.009%), isliye sirf BCE kaafi nahi hota.

    English: The Dice term is computed per sample and averaged, so a large
    defect does not dominate the loss of the small ones in the same batch.
    Hindi: Dice term har sample ke liye alag compute hota hai, isliye batch
    mein koi bada defect chhote defects ko dominate nahi karta.
    """

    def __init__(self, smooth: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        return bce_loss + self.dice_weight * dice_loss(
            torch.sigmoid(logits), targets, self.smooth
        )


def tversky_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """English: Tversky loss penalizing false negatives (missed defect pixels) more.
    Hindi: Tversky loss jo false negatives (missed defect pixels) ko zyada penalize karta hai.
    """
    flat_p = probs.flatten(1)
    flat_t = targets.flatten(1)
    true_pos = (flat_p * flat_t).sum(dim=1)
    false_neg = (flat_t * (1.0 - flat_p)).sum(dim=1)
    false_pos = ((1.0 - flat_t) * flat_p).sum(dim=1)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1.0 - tversky.mean()


class FocalTverskyBCELoss(nn.Module):
    """English: Combined BCE + Focal Tversky Loss for extreme foreground imbalance.
    Hindi: BCE aur Focal Tversky loss ka combination jo extreme defect imbalance ko handle karta hai.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        flat_p = probs.flatten(1)
        flat_t = targets.flatten(1)
        true_pos = (flat_p * flat_t).sum(dim=1)
        false_neg = (flat_t * (1.0 - flat_p)).sum(dim=1)
        false_pos = ((1.0 - flat_t) * flat_p).sum(dim=1)
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        focal_tversky = torch.pow(1.0 - tversky, self.gamma).mean()
        return self.bce_weight * bce_loss + focal_tversky


@torch.no_grad()
def binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """English: Dice, IoU, precision and recall for a batch of masks.
    Hindi: Mask batch ke liye Dice, IoU, precision aur recall.

    English: Each sample is scored separately and then averaged, which is the
    correct approach for this dataset because a patch with zero ground-truth
    defect would otherwise contribute a misleading 1.0 to a Dice computed over
    the whole batch.
    Hindi: Har sample ka score alag calculate hota hai phir average hota hai,
    kyunki batch-level Dice mein zero-defect patch galat 1.0 de sakta hai.
    """
    predictions = (torch.sigmoid(logits) > threshold).float()

    flat_pred = predictions.flatten(1)
    flat_true = targets.flatten(1)

    intersection = (flat_pred * flat_true).sum(dim=1)
    predicted_sum = flat_pred.sum(dim=1)
    true_sum = flat_true.sum(dim=1)

    dice = (2 * intersection + 1.0) / (predicted_sum + true_sum + 1.0)

    union = predicted_sum + true_sum - intersection
    iou = (intersection + 1.0) / (union + 1.0)

    precision = (intersection + 1.0) / (predicted_sum + 1.0)
    recall = (intersection + 1.0) / (true_sum + 1.0)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }
