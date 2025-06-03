from lightning import LightningModule
import torch
from torch import Tensor
from torch.nn import functional, Module, CrossEntropyLoss
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.functional.segmentation as segmentation_metrics
import torchmetrics.functional as classification_metrics
import torchvision

from datasets.segmentation import CrackSeg
from train_utils import segmentation


class PAEDTrainer(LightningModule):

    def __init__(self, model: Module):
        super().__init__()
        self.model = model

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
        batch_size = prediction.size(0) # (B, 1, H, W)
        total_paed = torch.zeros(1, device=target.device, requires_grad=True)
        for b in range(batch_size):
            pred_mask_b = prediction[b].squeeze()
            pred_mask_b_skeletonize=CrackSeg.skeletonize(pred_mask_b)
            sdf_ext_b = sdf_ext[b].squeeze()
            sdf_int_b = sdf_int[b].squeeze()
            distances = torch.sum(sdf_ext_b*pred_mask_b_skeletonize - 10*sdf_int_b*pred_mask_b) # + (sdf_int_b - edt_map(pred_mask_b))
            # distances = torch.sum(sdf_ext_b*pred_mask_b - 10*sdf_int_b*pred_mask_b_skeletonize)
            total_paed = total_paed + distances
        return total_paed / batch_size

    def _forward_step(self, batch, batch_idx):
        return self._forward_step_paed(batch, batch_idx)

    def _forward_step_joint_loss(self, batch, batch_idx):
        images, masks = batch
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
        outputs = self.model(images)
        predictions = functional.sigmoid(outputs)
        loss = self.paed_loss(predictions, masks, sdf_ext, sdf_int)
        preds = (predictions.detach() > 0.5).int()
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


class TCNClassificationLit(LightningModule):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx) -> Tensor:
        loss, accuracy = self._forward_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "train_acc": accuracy})
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        loss, accuracy = self._forward_step(batch, batch_idx)
        self.log_dict({"val_loss": loss, "val_acc": accuracy})
        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        loss, accuracy = self._forward_step(batch, batch_idx)
        self.log_dict({"test_loss": loss, "test_acc": accuracy})
        return loss

    def forward(self, x) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, patience=20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _forward_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = functional.cross_entropy(outputs, labels)
        aux_output = outputs.detach()
        accuracy = classification.classification_accuracy(labels, aux_output)
        return loss, accuracy