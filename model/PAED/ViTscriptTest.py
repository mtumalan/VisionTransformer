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

def get_bounding_boxes(binary_mask):
    labeled_array, num_features = label(binary_mask)
    boxes = []
    for region_label in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == region_label)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        boxes.append((y_min, x_min, y_max, x_max))
    return boxes

# --- Setup ---
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
classdict_path = cwd + '/../VisionChallenge/collaboration_it_mx/output_images/calss_names_colors.csv'
print(classdict_path)

rgb_to_class, class_names = load_classdict(classdict_path)
num_classes = len(class_names)

#train_path  = cwd + '/../VisionChallenge/Attachments/Attachments'
train_path  = cwd + '/../VisionChallenge/Attachments/shifted'
print(train_path)

image_dir = os.path.join(train_path, 'image_png')
mask_dir = os.path.join(train_path, 'mask_png')

image_filenames = os.listdir(image_dir)
mask_filenames = os.listdir(mask_dir)

train_images, temp_images, train_masks, temp_masks = train_test_split(
    image_filenames, mask_filenames, test_size=0.3, random_state=42
)

valid_images, test_images, valid_masks, test_masks = train_test_split(
    temp_images, temp_masks, test_size=0.5, random_state=42
)

print(f"Train: {len(train_images)}, Validation: {len(valid_images)}, Test: {len(test_images)}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = StructuralDamageDataset(image_dir, mask_dir, classdict_path, transform=transform)
valid_dataset = StructuralDamageDataset(image_dir, mask_dir, classdict_path, transform=transform)
test_dataset = StructuralDamageDataset(image_dir, mask_dir, classdict_path, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, persistent_workers=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, persistent_workers=True)

# Instantiate
configurations = [
    {'ID': 0, 'patch_size': 16, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 1, 'patch_size': 16, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 2, 'patch_size': 16, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
    {'ID': 3, 'patch_size': 8, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 4, 'patch_size': 8, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 5, 'patch_size': 8, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
    
    {'ID': 6, 'patch_size': 4, 'hidden_size': 512, 'hidden_layers': 8, 'attention_heads': 8},
    {'ID': 7, 'patch_size': 4, 'hidden_size': 768, 'hidden_layers': 12, 'attention_heads': 12},
    {'ID': 8, 'patch_size': 4, 'hidden_size': 1024, 'hidden_layers': 16, 'attention_heads': 16},
]

for config in configurations:
    version_n = config['ID']
    patch_size = config['patch_size']
    hidden_size = config['hidden_size']
    hidden_layers = config['hidden_layers']
    attention_heads = config['attention_heads']
    print(f"Testing Version {version_n}: Patch Size {patch_size}, Hidden Size {hidden_size}, Hidden Layers {hidden_layers}, Attention Heads {attention_heads}")
    '''
    model = LightningViTModel(
        num_classes=num_classes,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=hidden_layers,
        num_attention_heads=attention_heads
    )
'''
    model = PAEDTrainer(num_classes=num_classes, patch_size = 16, hidden_size = 512,num_hidden_layers = 8,num_attention_heads = 8)
    # Definir ruta del checkpoint
    checkpoint_path = get_latest_checkpoint(version_n, cwd)

    # Directorio de checkpoints
    lv_checkpoint = os.path.join(cwd, f'logs/vit-model/version_{version_n}/checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=lv_checkpoint)
    match = re.search(r'epoch=(\d+)', checkpoint_path)
    epoch = int(match.group(1)) if match else None

    # Definir el trainer
    trainer = L.Trainer(callbacks=[checkpoint_callback], num_sanity_val_steps=0, max_epochs=epoch)

    # Entrenamiento
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=checkpoint_path)

    num_batches =  10#125 #len(test_data_loader) #10 de 50 a 125

    model_name = f'ID{version_n}P{patch_size}H{hidden_size}A{attention_heads}'
    output_dir = os.path.join(cwd, 'test', model_name)
    os.makedirs(output_dir, exist_ok=True)

    index_to_class = {i: name for i, name in enumerate(class_names)}

    def get_present_classes(mask, index_to_class):
        unique_classes = np.unique(mask)
        class_names = [f"{c}: {index_to_class.get(c, 'Unknown')}" for c in unique_classes]
        return class_names

    def dice_coefficient(gt, pred, class_idx):
        gt_bin = (gt == class_idx)
        pred_bin = (pred == class_idx)
        intersection = np.logical_and(gt_bin, pred_bin).sum()
        size_sum = gt_bin.sum() + pred_bin.sum()
        if size_sum == 0:
            return float('nan')
        return 2 * intersection / size_sum

    batch_iterator = iter(test_dataloader)

    csv_path = os.path.join(output_dir, f'{model_name}_metrics.csv')

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Model_ID", "Model_Name", "Patch_Size", "Hidden_Size", "Layers", "Heads",
            "Batch_Num", "Image_Idx",
            "Accuracy", "Mean_IoU", "Mean_Dice", "Inference_Time",
            "GT_Classes", "Pred_Classes", "Missing_Classes", "False_Positive_Classes"
        ])

        for batch_num in range(num_batches):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break

            start_time = time.time()
            with torch.no_grad():
                model.eval()
                logits = model(batch[0])
            pr_masks = logits.sigmoid()
            inference_time = time.time() - start_time
            avg_time_per_image = inference_time / len(batch[0])

            for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch[0], batch[1], pr_masks)):
                gt_mask_np = gt_mask.numpy().squeeze().astype(np.uint8)
                pred_labels = pr_mask.argmax(dim=0).cpu().numpy()
                gt_resized = np.array(Image.fromarray(gt_mask_np).resize(pred_labels.shape[::-1], resample=Image.NEAREST))

                comparison = (gt_resized != pred_labels).astype(float)
                num_mismatches = int(comparison.sum())
                total_pixels = comparison.size
                accuracy = 100 * (1 - num_mismatches / total_pixels)

                ious = []
                dices = []
                for class_idx in range(num_classes):
                    gt_bin = (gt_resized == class_idx)
                    pred_bin = (pred_labels == class_idx)
                    intersection = np.logical_and(gt_bin, pred_bin).sum()
                    union = np.logical_or(gt_bin, pred_bin).sum()
                    iou = float('nan') if union == 0 else intersection / union
                    dice = dice_coefficient(gt_resized, pred_labels, class_idx)

                    ious.append(iou)
                    dices.append(dice)

                mean_iou = np.nanmean(ious)
                mean_dice = np.nanmean(dices)

                gt_classes = sorted(set(int(c) for c in np.unique(gt_resized)))
                pred_classes = sorted(set(int(c) for c in np.unique(pred_labels)))
                missing_classes = sorted(set(gt_classes) - set(pred_classes))
                false_positive_classes = sorted(set(pred_classes) - set(gt_classes))

                writer.writerow([
                    version_n, model_name, patch_size, hidden_size, hidden_layers, attention_heads,
                    batch_num, idx,
                    accuracy, mean_iou, mean_dice, avg_time_per_image,
                    "|".join(map(str, gt_classes)),
                    "|".join(map(str, pred_classes)),
                    "|".join(map(str, missing_classes)),
                    "|".join(map(str, false_positive_classes))
                ])

            # Visualización por batch e imagen
            if batch_num <= num_batches/5: #25:
                for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch[0], batch[1], pr_masks)):
                    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
                    ax1, ax2, ax3, ax4, ax5 = axes
                    fig.suptitle(f'Model: {model_name} - Batch {batch_num} - Image {idx}', fontsize=14)

                    img_np = image.numpy().transpose(1, 2, 0)
                    ax1.imshow(img_np)
                    ax1.set_title("Image")
                    ax1.axis("off")

                    index_to_color = np.zeros((num_classes, 3), dtype=np.uint8)
                    for rgb, class_index in rgb_to_class.items():
                        index_to_color[class_index] = list(rgb)

                    gt_mask_np = gt_mask.numpy().squeeze().astype(np.uint8)
                    colored_gt = index_to_color[gt_mask_np]
                    ax2.imshow(colored_gt)
                    ax2.set_title("Ground truth")
                    ax2.axis("off")

                    gt_classes_idx = np.unique(gt_mask_np)
                    for i, class_idx in enumerate(gt_classes_idx):
                        class_name = index_to_class[class_idx]
                        color = index_to_color[class_idx] / 255.0
                        y_pos = 0.98 - i * 0.05

                        ax2.add_patch(plt.Rectangle((0.01, y_pos - 0.02), 0.03, 0.025, transform=ax2.transAxes,
                                                    color=color, clip_on=False))
                        ax2.text(0.05, y_pos, f"{class_idx}: {class_name}", transform=ax2.transAxes,
                                fontsize=8, va='top', ha='left', color='white',
                                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

                    pred_labels = pr_mask.argmax(dim=0).cpu().numpy()
                    colored_pred = index_to_color[pred_labels]
                    ax3.imshow(colored_pred)
                    ax3.set_title("Prediction")
                    ax3.axis("off")

                    pr_classes_idx = np.unique(pred_labels)
                    for i, class_idx in enumerate(pr_classes_idx):
                        class_name = index_to_class[class_idx]
                        color = index_to_color[class_idx] / 255.0
                        y_pos = 0.98 - i * 0.05

                        ax3.add_patch(plt.Rectangle((0.01, y_pos - 0.02), 0.03, 0.025, transform=ax3.transAxes,
                                                    color=color, clip_on=False))
                        ax3.text(0.05, y_pos, f"{class_idx}: {class_name}", transform=ax3.transAxes,
                                fontsize=8, va='top', ha='left', color='white',
                                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

                    gt_resized = np.array(Image.fromarray(gt_mask_np).resize((224, 224), resample=Image.NEAREST))
                    comparison = (gt_resized != pred_labels).astype(float)
                    mismatch_cmap = ListedColormap(["white", "red"])

                    ax4.imshow(comparison, cmap=mismatch_cmap, interpolation='none')
                    ax4.set_title("Mismatch Highlight")
                    ax4.axis("off")

                    num_mismatches = int(comparison.sum())
                    total_pixels = comparison.size
                    accuracy = 100 * (1 - num_mismatches / total_pixels)
                    ious = []
                    for class_idx in range(num_classes):
                        gt_bin = (gt_resized == class_idx)
                        pred_bin = (pred_labels == class_idx)

                        intersection = np.logical_and(gt_bin, pred_bin).sum()
                        union = np.logical_or(gt_bin, pred_bin).sum()

                        iou = float('nan') if union == 0 else intersection / union
                        ious.append(iou)

                    mean_iou = np.nanmean(ious)

                    info_text = f"Errores: {num_mismatches}  ({accuracy:.1f}%)\nmIoU: {mean_iou*100:.1f}%"
                    ax4.text(0.5, -0.08, info_text,
                            transform=ax4.transAxes, ha='center', fontsize=8, color='blue',
                            bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'))

                    ax5.imshow(img_np)
                    ax5.set_title("Predicted Regions with Boxes")
                    ax5.axis("off")

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
                            ax5.add_patch(rect)
                            ax5.text(x_min, y_min - 3, f"{index_to_class[class_idx]}",
                                    color=color, fontsize=8, weight='bold',
                                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    output_path = os.path.join(output_dir, f'result_batch{batch_num}_img{idx}.png')
                    plt.savefig(output_path, bbox_inches='tight')
                    plt.close()

    # metricas de training
    metrics_path = os.path.join(cwd, f'logs/vit-model/version_{version_n}', 'metrics.csv')
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        df_epochs = metrics.groupby('epoch').mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f'Model: {model_name}', fontsize=14) 
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Values')

        ax.plot(df_epochs['train_loss_epoch'], label="Training Loss")
        ax.plot(df_epochs['valid_loss'], label="Validation Loss")
        ax.plot(df_epochs['train_iou_epoch'], label="Training IoU")
        #if df_epochs['train_paed_epoch']:
        #    ax.plot(df_epochs['train_paed_epoch'], label="Training PAED")

        ax.set_title("Training and Validation Metrics")
        ax.legend(loc='upper right')

        metrics_output_path = os.path.join(output_dir, 'metrics.png')
        plt.savefig(metrics_output_path)
        plt.close()
    else:
        print(f"metrics.csv not found in {metrics_path}")