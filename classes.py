import os, glob, random, shutil, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
#import timm
import torch.nn as nn

import torchvision
from torchvision import transforms

import segmentation_models_pytorch as smp

import lightning as L

import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class StructuralDamageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, classdict_path=None, transform=None, target_transform=None, lazy_class_mapping=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.classdict_path = classdict_path
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.target_transform = target_transform
        self.lazy_class_mapping = lazy_class_mapping  # Option to defer class mapping

        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks must be equal!")

        if not lazy_class_mapping:
            # Process all masks to build class mapping
            self._build_class_mapping()
        else:
            self.unique_values = None  # Will be lazily built

    def _build_class_mapping(self):
        # Efficiently compute unique values across the dataset
        all_values = set()
        for mask_file in self.masks:
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = np.array(Image.open(mask_path).convert('L'))
            all_values.update(np.unique(mask))

        self.unique_values = sorted(all_values)  # Ensure consistent ordering
        self.value_to_class = {v: i for i, v in enumerate(self.unique_values)}
        self.num_classes = len(self.unique_values)

    def __len__(self):
        return len(self.images)

    def _lazy_class_mapping(self):
        if self.unique_values is None:
            self._build_class_mapping()

    def __getitem__(self, idx):
        try:
            # Build the class mapping lazily if needed
            if self.lazy_class_mapping:
                self._lazy_class_mapping()

            # Get paths
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])

            # Load the image and mask
            image = Image.open(img_path).convert('RGB')  # RGB image
            mask = Image.open(mask_path).convert('L')    # Grayscale mask (class indices)

            # Resize mask
            mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask_np = np.array(mask, dtype=np.int64)

            # Remap mask values to class indices
            mask_mapped = np.vectorize(self.value_to_class.get)(mask_np)

            # Convert to one-hot encoding
            mask_onehot = np.zeros((self.num_classes, mask_np.shape[0], mask_np.shape[1]), dtype=np.float32)
            for class_idx in range(self.num_classes):
                mask_onehot[class_idx][mask_mapped == class_idx] = 1.0

            # Convert mask to tensor
            mask_tensor = torch.tensor(mask_mapped, dtype=torch.long)

            # Apply transformations
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            if self.target_transform:
                mask_tensor = self.target_transform(mask_tensor)

            return image, mask_tensor
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise e
        
class StructuralDamageModel(L.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes):
        super().__init__()

        # Initialize the segmentation model
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, encoder_weights=None)

        # Store losses
        self.losses = {
            'valid': [],
            'train': [],
            'test': []
        }

        # Preprocessing parameters
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # CrossEntropyLoss for multiclass segmentation
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]  # Input image: [batch_size, num_channels, height, width]
        mask = batch[1]   # Target mask: [batch_size, height, width] (class indices)

        logits_mask = self.forward(image)  # Model output: [batch_size, num_classes, height, width]

        # Compute CrossEntropyLoss directly using class indices
        loss = self.loss_fn(logits_mask, mask)

        # Metrics
        prob_mask = logits_mask.softmax(dim=1)  # Apply softmax to logits
        pred_mask = prob_mask.argmax(dim=1)  # Shape: [batch_size, height, width]
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, mask, mode="multiclass", num_classes=logits_mask.shape[1])
        
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.losses[stage].append(loss)

        return loss

    def compute_metrics(self, tp, fp, fn, tn):
        # Sumar las métricas a lo largo de las dimensiones adecuadas
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()

        # Calcular el total
        total = tp + fp + fn + tn
        
        # Evitar la división por cero
        accuracy = (tp + tn) / total if total != 0 else torch.tensor(0.0, device=tp.device)
        recall = tp / (tp + fn) if (tp + fn) != 0 else torch.tensor(0.0, device=tp.device)
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else torch.tensor(0.0, device=tp.device)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else torch.tensor(0.0, device=tp.device)

        return accuracy, recall, f1_score

    def shared_epoch_end(self, stage):
        # Aggregate metrics
        tp = self.tp
        fp = self.fp
        fn = self.fn
        tn = self.tn

        # Compute IoU metrics
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Compute Accuracy, Recall, and F1 Score
        accuracy, recall, f1_score = self.compute_metrics(tp, fp, fn, tn)

        # Log metrics
        metrics = {
            f"{stage}_loss": torch.stack(self.losses[stage]).mean(),
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": accuracy,
            f"{stage}_recall": recall,
            f"{stage}_f1_score": f1_score,
        }

        self.log_dict(metrics, prog_bar=True, logger=True)

    def training_step(self, batch):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

class PAEDLoss(torch.nn.Module):
    def forward(self, preds, targets):
        abs_diff = torch.abs(preds - targets)
        paed = torch.mean(abs_diff / (targets + 1e-6))  
        return paed

class ViTSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        config = ViTConfig( #supuestos params de vit base patch16 224
        image_size=224,             # Tamaño de imagen esperado
        patch_size=8,              # Tamaño de cada patch (16x16 px) -
        num_channels=3,             # RGB
        hidden_size=768,            # Dimensión del embedding por patch -
        num_hidden_layers=12,       # Número de bloques Transformer -
        num_attention_heads=12,     # Número de "cabezas" de atención -
        intermediate_size=3072,     # Dimensión del feedforward interno
        qkv_bias=True,              # Usar sesgo en QKV lineales (como ViT original)
        hidden_dropout_prob=0.1,    # Dropout entre bloques --
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        )

        self.backbone = ViTModel(config)
        #self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, features_only=True)
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.backbone.config.hidden_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        outputs = self.backbone(x)
        hidden_states = outputs.last_hidden_state  # [batch, num_patches+1, hidden_size]

        hidden_states = hidden_states[:, 1:, :]  # rCLS token

        batch_size, num_patches, hidden_size = hidden_states.shape
        h = w = int(num_patches ** 0.5)  # ejemplo: 14x14 para imagen 224 con patch 16

        features = hidden_states.transpose(1, 2).reshape(batch_size, hidden_size, h, w)

        out = self.seg_head(features)

        # Upsample back to original input size
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out

# Lightning Moduleclass 
class LightningViTModel(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = ViTSegmentationModel(num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.paed_loss = PAEDLoss()

    def forward(self, x):
        return self.model(x)

    def _resize_target(self, y, size):
        return F.interpolate(y.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = self._resize_target(y, size=(224, 224))
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)  # Predicciones finales

        paed = self.paed_loss(preds.float(), y.float())
        #paed = torch.tensor(0.123) #constante test 
        self.log("train_paed", paed, prog_bar=True, on_epoch=True, logger=True)        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = self._resize_target(y, size=(224, 224))
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        paed = self.paed_loss(preds.float(), y.float())
        #paed = torch.tensor(0.456) #constante test
        self.log("val_paed", paed, prog_bar=True, on_epoch=True, logger=True)
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = self._resize_target(y, size=(224, 224))
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)