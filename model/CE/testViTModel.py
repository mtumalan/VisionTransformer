from io import BytesIO
import PIL.Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision import transforms
import os
import re
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classes import StructuralDamageDataset, ViTSegmentationModel, LightningViTModel
from functions import load_classdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import ModelCheckpoint
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import label
import time
import csv
# Entrada esperada
def get_bounding_boxes(binary_mask):
    labeled_array, num_features = label(binary_mask)
    boxes = []
    for region_label in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == region_label)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        boxes.append((y_min, x_min, y_max, x_max))
    return boxes

def get_latest_checkpoint(version_n: int, base_path: str) -> Optional[str]:
    checkpoint_dir = os.path.join(base_path, f'logs/vit-model/version_{version_n}/checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} does not exist.")
        return None

    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

    if not ckpt_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None

    latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('=')[1].split('-')[0]))

    latest_ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    print(f"Latest checkpoint found: {latest_ckpt_path}")
    return latest_ckpt_path

cwd = os.getcwd()
classdict_path = cwd + '/../../VisionChallenge/collaboration_it_mx/output_images/calss_names_colors.csv'
print(classdict_path)

rgb_to_class, class_names = load_classdict(classdict_path)
num_classes = len(class_names)

imagen = PIL.Image.open("IMG_3305.jpg")  # o cualquier forma de entrada válida
modelo = [0, 5] #modelos a testear
result_images = []
configurations = [
    {'ID': 0, 'patch_size': 16, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 1, 'patch_size': 16, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 2, 'patch_size': 16, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
    {'ID': 3, 'patch_size': 8, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 4, 'patch_size': 8, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 5, 'patch_size': 8, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
    {'ID': 6, 'patch_size': 4, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 7, 'patch_size': 4, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 8, 'patch_size': 4, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
]
# Filtro de configuraciones
if isinstance(modelo, int):
    modelo = [modelo]
elif modelo == '' or modelo is None:
    modelo = None  

if modelo is not None:
    configurations = [conf for conf in configurations if conf['ID'] in modelo]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_tensor = transform(imagen).unsqueeze(0)  # Añade batch dim: [1, C, H, W]

for config in configurations:
    version_n = config['ID']
    patch_size = config['patch_size']
    hidden_size = config['hidden_size']
    hidden_layers = config['hidden_layers']
    attention_heads = config['attention_heads']

    print(f"Evaluando modelo {version_n}")

    # Cargar modelo
    model = LightningViTModel(
        num_classes=num_classes,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=hidden_layers,
        num_attention_heads=attention_heads
    )

    checkpoint_path = get_latest_checkpoint(version_n, cwd)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()

    # Inferencia
    with torch.no_grad():
        logits = model(image_tensor)
        pr_mask = logits.sigmoid().squeeze(0)  # shape [num_classes, H, W]

    pred_labels = pr_mask.argmax(dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    ax1, ax2, ax3, ax4 = axes
    model_name = f'P{patch_size}H{hidden_size}A{attention_heads}'

    # Imagen original
    img_np = np.array(imagen.resize((224, 224)))
    ax1.imshow(img_np)
    ax1.set_title("Original")
    ax1.axis("off")
    index_to_class = {i: name for i, name in enumerate(class_names)}
    # Predicción
    index_to_color = np.zeros((num_classes, 3), dtype=np.uint8)
    for rgb, class_index in rgb_to_class.items():
        index_to_color[class_index] = list(rgb)

    colored_pred = index_to_color[pred_labels]
    ax2.imshow(colored_pred)
    ax2.set_title("Prediction")
    ax2.axis("off")

    # Clases predichas
    pr_classes_idx = np.unique(pred_labels)
    for i, class_idx in enumerate(pr_classes_idx):
        class_name = index_to_class[class_idx]
        color = index_to_color[class_idx] / 255.0
        y_pos = 0.98 - i * 0.05
        ax2.add_patch(plt.Rectangle((0.01, y_pos - 0.02), 0.03, 0.025, transform=ax2.transAxes,
                                    color=color, clip_on=False))
        ax2.text(0.05, y_pos, f"{class_idx}: {class_name}", transform=ax2.transAxes,
                 fontsize=8, va='top', ha='left', color='white',
                 bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    # Mismatches vs ground truth (dummy si no tienes GT)
    comparison = np.zeros_like(pred_labels, dtype=np.uint8)
    mismatch_cmap = ListedColormap(["white", "red"])
    ax3.imshow(comparison, cmap=mismatch_cmap)
    ax3.set_title("Mismatch (placeholder)")
    ax3.axis("off")

    # Boxes
    ax4.imshow(img_np)
    ax4.set_title("Predicted Regions with Boxes")
    ax4.axis("off")
    for class_idx in pr_classes_idx:
        if class_idx == 0:
            continue
        binary_mask = (pred_labels == class_idx)
        boxes = get_bounding_boxes(binary_mask)
        color = index_to_color[class_idx] / 255.0
        for (y_min, x_min, y_max, x_max) in boxes:
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            rect = plt.Rectangle((x_min, y_min), width, height,
                                 edgecolor=color, facecolor='none', linewidth=2)
            ax4.add_patch(rect)
            ax4.text(x_min, y_min - 3, f"{index_to_class[class_idx]}",
                     color=color, fontsize=8, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    # Guardar en memoria
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_result = PIL.Image.open(buf).convert("RGB")
    result_images.append({
        'model_name': model_name,
        'image': image_result
    })
    plt.close()


result_images[0]['image'].show() #ver resultado primer modelo parametros
