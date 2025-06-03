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

from torch import Tensor
from torch.nn import functional, Module, CrossEntropyLoss
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.functional.segmentation as segmentation_metrics
import torchmetrics.functional as classification_metrics

from segmentation import CrackSeg
import segmentation

import numpy as np
#from skimage.morphology import skeletonize as skimage_skeletonize


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
            sdf_ext, sdf_int = segmentation.compute_sdf(mask)
            # Convert mask to tensor
            mask_tensor = torch.tensor(mask_mapped, dtype=torch.long)
            sdf_ext = torch.tensor(sdf_ext).float()
            sdf_int = torch.tensor(sdf_int).float()
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            if self.target_transform:
                mask_tensor = self.target_transform(mask_tensor)

            return image, mask_tensor, sdf_ext, sdf_int
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
    
"""
def paed_loss(msk, pred_mask, threshold=0.35):
    batch_size = msk.size(0)
    total_paed = 0.0  # acumular como float para no romper gráfica

    for b in range(batch_size):
        msk_b = msk[b].squeeze()
        pred_mask_b = pred_mask[b].squeeze()

        coord_msk = (msk_b >= threshold).nonzero(as_tuple=True)
        coord_pred = (pred_mask_b >= threshold).nonzero(as_tuple=True)

        x_coord_msk = coord_msk[0].float()
        y_coord_msk = coord_msk[1].float()
        x_coord_pred = coord_pred[0].float()
        y_coord_pred = coord_pred[1].float()

        n = len(x_coord_msk)
        m = len(x_coord_pred)

        if n == 0 and m == 0:
            paed = torch.tensor(0.0, device=msk.device)
        elif n == 0:
            distance_S2_S1 = torch.sum(torch.sqrt(x_coord_pred**2 + y_coord_pred**2))
            paed = distance_S2_S1 / m
        elif m == 0:
            distance_S1_S2 = torch.sum(torch.sqrt(x_coord_msk**2 + y_coord_msk**2))
            paed = distance_S1_S2 / n
        else:
            mask_coordinates = torch.stack((x_coord_msk, y_coord_msk), dim=1)
            prediction_coordinates = torch.stack((x_coord_pred, y_coord_pred), dim=1)

            distance_x1i_S2 = torch.zeros(n, device=msk.device)
            for i in range(n):
                distances = torch.sqrt(torch.sum((mask_coordinates[i] - prediction_coordinates) ** 2, dim=1))
                distance_x1i_S2[i] = torch.min(distances)

            distance_S1_S2 = torch.sum(distance_x1i_S2)

            distance_x2j_S1 = torch.zeros(m, device=msk.device)
            for j in range(m):
                distances = torch.sqrt(torch.sum((prediction_coordinates[j] - mask_coordinates) ** 2, dim=1))
                distance_x2j_S1[j] = torch.min(distances)

            distance_S2_S1 = torch.sum(distance_x2j_S1)

            paed = (distance_S1_S2 + distance_S2_S1 + 0.001) / (n + m + 0.001)

        total_paed += paed

    return total_paed / batch_size
"""
def paed_loss_multiclass_soft(msk, pred_mask, num_classes=17, sigma=3, class_penalty=True):

    batch_size, C, H, W = msk.shape
    device = msk.device

    size = int(6 * sigma + 1)
    x = torch.arange(size).float() - size // 2
    gauss = torch.exp(- (x ** 2) / (2 * sigma ** 2))
    gauss_kernel = gauss[:, None] * gauss[None, :]
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    gauss_kernel = gauss_kernel.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # Expandir kernel para conv2d con grupos = C
    gauss_kernel = gauss_kernel.repeat(C, 1, 1, 1)  # [C,1,H,W]

    # Suavizado
    msk_smooth = F.conv2d(msk, gauss_kernel, padding=size//2, groups=C)
    pred_smooth = F.conv2d(pred_mask, gauss_kernel, padding=size//2, groups=C)

    # Penalización básica: diferencia absoluta entre máscaras suavizadas
    base_loss = torch.abs(msk_smooth - pred_smooth)

    if class_penalty:
        # Penalizar más cuando se predice probabilidad en una clase incorrecta
        class_mismatch = msk * (1 - pred_mask)  # donde predice mal, da valores altos
        penalty_map = class_mismatch * base_loss * 2
        dist = penalty_map.mean(dim=[2, 3])  # Promedio espacial
    else:
        dist = base_loss.mean(dim=[2, 3])  # Promedio espacial

    loss_per_sample = dist.mean(dim=1)  # Promedio por clase
    total_loss = loss_per_sample.mean()  # Promedio por batch

    return total_loss

    
class ViTSegmentationModel(nn.Module):
    def __init__(self, num_classes, patch_size, hidden_size,num_hidden_layers,num_attention_heads):
        super().__init__()
        config = ViTConfig( #supuestos params de vit base patch16 224
        image_size=224,             # Tamaño de imagen esperado
        patch_size=patch_size,              # Tamaño de cada patch (16x16 px) -
        num_channels=3,             # RGB
        hidden_size=hidden_size,            # Dimensión del embedding por patch -
        num_hidden_layers=num_hidden_layers,       # Número de bloques Transformer -
        num_attention_heads=num_attention_heads,     # Número de "cabezas" de atención -
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
class LightningViTModel(L.LightningModule):
    def __init__(self, num_classes, patch_size, hidden_size,num_hidden_layers,num_attention_heads):
        super().__init__()
        num_classes = 17
        self.num_classes = num_classes
        #self.num_classes = 17
        self.model = ViTSegmentationModel(num_classes, patch_size, hidden_size,num_hidden_layers,num_attention_heads)
        # Usamos PAED como loss, pasando num_classes
       #self.loss_fn = paed_loss_multiclass(thres=0.6, thres2=0.6, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _resize_target(self, y, size):
        return F.interpolate(y.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()
    def iou_score(self, preds, targets, num_classes = 17):
            # Convertir a one-hot encoding
            preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Iterar por cada clase para calcular el IoU
            iou_per_class = []
            for c in range(num_classes):
                preds_c = preds_one_hot[:, c, :, :]
                targets_c = targets_one_hot[:, c, :, :]

                intersection = (preds_c * targets_c).sum((1, 2))
                union = (preds_c + targets_c).clamp(0, 1).sum((1, 2))

                iou_c = (intersection + 1e-6) / (union + 1e-6)
                iou_per_class.append(iou_c.mean())

            return torch.tensor(iou_per_class).mean()
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = self._resize_target(y, size=(224, 224))
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        print(y.min(), y.max(), y.shape, self.num_classes)
        y = y.long()
        y_one_hot = F.one_hot(y, self.num_classes).permute(0, 3, 1, 2).float()


        loss = paed_loss_multiclass_soft(y_one_hot, probs, num_classes=self.num_classes)

        # También puedes calcular IOU para monitoreo
        iou = self.iou_score(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_iou", iou, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = self._resize_target(y, size=(224, 224))

        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        print(y.min(), y.max(), y.shape, self.num_classes)
        y = y.long()
        y_one_hot = F.one_hot(y, self.num_classes).permute(0, 3, 1, 2).float()  # convierte y a one-hot [B,C,H,W]

        loss = paed_loss_multiclass_soft(y_one_hot, probs, num_classes=self.num_classes)
        iou = self.iou_score(preds, y)

        self.log("valid_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("valid_iou", iou, prog_bar=True, on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class PAEDTrainer(L.LightningModule):

    def __init__(self, num_classes, patch_size, hidden_size,num_hidden_layers,num_attention_heads):
        super().__init__()
        self.model = ViTSegmentationModel(num_classes, patch_size, hidden_size,num_hidden_layers,num_attention_heads)
        #self.model = model
    def _resize_target(self, y, size):
        return F.interpolate(y.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()
    def training_step(self, batch, batch_idx) -> Tensor:
        loss, accuracy, iou, dice, precision, recall, _ = self._forward_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "train_acc": accuracy, "train_IoU": iou, "train_recall": recall, "train_dice": dice, "train_precision": precision}, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        loss, accuracy, iou, dice, precision, recall, grid = self._forward_step(batch, batch_idx)
        self.log_dict({"val_loss": loss, "val_acc": accuracy, "val_IoU": iou, "val_recall": recall, "val_dice": dice, "val_precision": precision}, on_epoch=True)
        if grid is not None:
            self.logger.experiment.add_image('predictions', grid, self.global_step)
        return loss

    def test_step(self, batch, batch_idx):
        _, accuracy, iou, dice, precision, recall, grid = self._forward_step(batch, batch_idx)
        metrics = {
            "test_acc": accuracy, "test_IoU": iou, "test_recall": recall, "test_dice": dice, "test_precision": precision}
        self.log_dict(metrics, on_epoch=True)
        if grid is not None:
            self.logger.experiment.add_image('predictions', grid, self.global_step)
        return metrics

    def forward(self, x) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-4)
        # optimizer = SGD(self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, patience=30)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_IoU",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def paed_loss(self, prediction: Tensor, target: Tensor, sdf_ext: Tensor, sdf_int: Tensor):
        batch_size = prediction.size(0)  # (B, 1, H, W)
        total_paed = torch.zeros(1, device=target.device, requires_grad=True)

        for b in range(batch_size):
            pred_mask_b = prediction[b].squeeze()  # (H_pred, W_pred)
            pred_mask_b_skeletonize = CrackSeg.skeletonize(pred_mask_b)

            # SDFs originales (probablemente con resolución distinta, e.g., 1024x1024)
            sdf_ext_b = sdf_ext[b].squeeze()  # (H_sdf, W_sdf)
            sdf_int_b = sdf_int[b].squeeze()

            # Redimensionar SDFs al tamaño del pred_mask_b
            target_shape = pred_mask_b.shape[-2:]  # (H_pred, W_pred)
            sdf_ext_b = F.interpolate(sdf_ext_b.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze()
            sdf_int_b = F.interpolate(sdf_int_b.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze()

            # PAED Loss
            distances = torch.sum(sdf_ext_b * pred_mask_b_skeletonize - 10 * sdf_int_b * pred_mask_b)
            total_paed = total_paed + distances

        return total_paed / batch_size


    def _forward_step(self, batch, batch_idx):
        return self._forward_step_paed(batch, batch_idx)

    def _forward_step_joint_loss(self, batch, batch_idx):
        images, masks = batch
        masks = self._resize_target(masks, size=(224, 224))
        outputs = self.model(images)
        criterion = segmentation.JointDiceBCEWithLogitsLoss(0.8)
        loss = criterion(outputs, masks)
        predictions = functional.sigmoid(outputs.detach())
        preds = (predictions > 0.5).int()
        accuracy = segmentation.pixel_accuracy(masks, preds)
        iou = segmentation_metrics.mean_iou(preds, masks.int(), num_classes=1,
                                            include_background=False, per_class=False).mean()
        dice = segmentation_metrics.dice_score(preds, masks.int(), num_classes=1,
                                               include_background=False, average='macro').mean()
        precision = classification_metrics.precision(preds, masks.int(), task='binary',
                                                     multidim_average='global')
        recall = classification_metrics.recall(preds, masks.int(), task='binary',
                                               multidim_average='global')
        if iou.isnan().item():
            iou = torch.zeros(1, device=iou.device)
        grid = None
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(predictions)
        return loss, accuracy, iou, dice, precision, recall, grid

    def _forward_step_paed(self, batch, batch_idx):
        images, masks, sdf_ext, sdf_int = batch
        masks = self._resize_target(masks, size=(224, 224))
        outputs = self.model(images)
        predictions = functional.sigmoid(outputs)
        loss = self.paed_loss(predictions, masks, sdf_ext, sdf_int)
        preds = (predictions.detach() > 0.5).int()
        print(masks.shape)
        print(preds.shape)
        accuracy = segmentation.pixel_accuracy(masks, preds)
        iou = segmentation_metrics.mean_iou(preds, masks.int(), num_classes=1,
                                            include_background=False, per_class=False).mean()
        dice = segmentation_metrics.dice_score(preds, masks.int(), num_classes=1,
                                               include_background=False, average='macro').mean()
        precision = classification_metrics.precision(preds, masks.int(), task='binary',
                                                     multidim_average='global')
        recall = classification_metrics.recall(preds, masks.int(), task='binary',
                                               multidim_average='global')
        if iou.isnan().item():
            iou = torch.zeros(1, device=iou.device)
        grid = None
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(predictions)
        return loss, accuracy, iou, dice, precision, recall, grid

    def _forward_step_likelihood(self, batch, batch_idx):
        images, masks = batch
        masks = masks.long().squeeze()
        outputs = self.model(images)
        criterion = CrossEntropyLoss()
        loss = criterion(outputs, masks)
        preds = torch.argmax(outputs.detach(), dim=1)
        accuracy = segmentation.pixel_accuracy(masks, preds)
        iou = segmentation_metrics.mean_iou(preds, masks.int(), num_classes=1,
                                            include_background=False, per_class=False).mean()
        dice = segmentation_metrics.dice_score(preds, masks.int(), num_classes=1,
                                               include_background=False, average='macro').mean()
        precision = classification_metrics.precision(preds, masks.int(), task='binary',
                                                     multidim_average='global')
        recall = classification_metrics.recall(preds, masks.int(), task='binary',
                                               multidim_average='global')
        if iou.isnan().item():
            iou = torch.zeros(1, device=iou.device)
        grid = None
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(preds.unsqueeze(1))
        return loss, accuracy, iou, dice, precision, recall, grid