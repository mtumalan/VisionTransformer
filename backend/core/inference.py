# inference/inference.py
"""
Single-model ViT inference utilities.
Loads ONE checkpoint from settings.VIT_CHECKPOINT and returns
   {"mask.png": <bytes>, "bbox.png": <bytes>}
for a given input image (bytes).
"""

import os
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
from PIL import Image as PILImage, ImageDraw
import torch
from torchvision import transforms
from scipy.ndimage import label
from django.conf import settings

from .classes import LightningViTModel
from .functions import load_classdict


# --------------------------------------------------------------------------- #
# Helper: bounding-box extraction                                             #
# --------------------------------------------------------------------------- #
def _get_bounding_boxes(binary_mask: np.ndarray) -> list[Tuple[int, int, int, int]]:
    labeled, n = label(binary_mask)
    boxes: list[Tuple[int, int, int, int]] = []
    for region in range(1, n + 1):
        coords = np.argwhere(labeled == region)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        boxes.append((y_min, x_min, y_max, x_max))
    return boxes


# --------------------------------------------------------------------------- #
# Hard-coded ViT hyper-parameters (update if your checkpoint differs)         #
# --------------------------------------------------------------------------- #
DEFAULT_VIT_CFG = dict(
    patch_size=4,
    hidden_size=1024,
    num_hidden_layers=16,
    num_attention_heads=16,
)


# --------------------------------------------------------------------------- #
# Main public entry-point                                                     #
# --------------------------------------------------------------------------- #
def run_segmentation_outputs(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (224, 224),
) -> Dict[str, bytes]:
    """
    Arguments
    ---------
    image_bytes : bytes
        Raw bytes of the uploaded RGB image.
    target_size : (H, W)
        Size to which the model expects its inputs (default 224×224).

    Returns
    -------
    dict
        {"mask.png": colored-mask PNG bytes,
         "bbox.png": overlay-with-boxes PNG bytes}
    """

    # --- class dict ---------------------------------------------------------
    rgb_to_class, class_names = load_classdict(settings.CLASSDICT_CSV)
    num_classes = len(class_names)

    # --- image preprocessing ------------------------------------------------
    pil_img = PILImage.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose(
        [transforms.Resize(target_size), transforms.ToTensor()]
    )
    image_tensor = transform(pil_img).unsqueeze(0)  # [1, 3, H, W]

    # --- model skeleton -----------------------------------------------------
    vit = LightningViTModel(
        num_classes=num_classes,
        patch_size=DEFAULT_VIT_CFG["patch_size"],
        hidden_size=DEFAULT_VIT_CFG["hidden_size"],
        num_hidden_layers=DEFAULT_VIT_CFG["num_hidden_layers"],
        num_attention_heads=DEFAULT_VIT_CFG["num_attention_heads"],
    )

    # --- load checkpoint ----------------------------------------------------
    ckpt_path = settings.VIT_CHECKPOINT
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vit.load_state_dict(ckpt["state_dict"])
    vit.eval()

    # --- forward pass -------------------------------------------------------
    with torch.no_grad():
        logits = vit(image_tensor)               # [1, C, H, W]
        pr_mask = logits.sigmoid().squeeze(0)    # [C, H, W]

    pred_labels = pr_mask.argmax(dim=0).cpu().numpy()  # [H, W]

    # --- build colored mask -------------------------------------------------
    index_to_color = np.zeros((num_classes, 3), dtype=np.uint8)
    for rgb, idx in rgb_to_class.items():
        index_to_color[idx] = list(rgb)

    colored_mask = index_to_color[pred_labels]            # [H, W, 3]
    mask_png = _pil_to_png_bytes(PILImage.fromarray(colored_mask))

    # --- draw bounding boxes overlay ---------------------------------------
    target_img = pil_img.resize(target_size)
    draw = ImageDraw.Draw(target_img)
    for class_idx in range(1, num_classes):  # skip background (idx 0)
        bin_mask = (pred_labels == class_idx).astype(np.uint8)
        if bin_mask.sum() == 0:
            continue
        for y0, x0, y1, x1 in _get_bounding_boxes(bin_mask):
            rgb = tuple(int(c) for c in index_to_color[class_idx])
            draw.rectangle([x0, y0, x1, y1], outline=rgb, width=2)

    bbox_png = _pil_to_png_bytes(target_img)

    return {"mask.png": mask_png, "bbox.png": bbox_png}


# --------------------------------------------------------------------------- #
# Utility: PIL → PNG bytes                                                    #
# --------------------------------------------------------------------------- #
def _pil_to_png_bytes(img: PILImage.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
