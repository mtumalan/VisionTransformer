import os
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

# --- Setup ---

cwd = os.getcwd()
classdict_path = cwd + '/../VisionChallenge/collaboration_it_mx/output_images/calss_names_colors.csv'
print(classdict_path)

rgb_to_class = load_classdict(classdict_path)
num_classes = len(rgb_to_class)

train_path  = cwd + '/../VisionChallenge/Attachments/Attachments'
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

model = LightningViTModel(num_classes=num_classes, patch_size = 4, hidden_size = 1024,num_hidden_layers = 16,num_attention_heads = 16)

#earlystop_callback = EarlyStopping(monitor="valid_loss", patience=3, verbose=True, mode="min")
#logger = CSVLogger(save_dir="logs/", name="vit-model")

#trainer = L.Trainer(
#    max_epochs=100,
#    logger=logger,
#    callbacks=[earlystop_callback],
#    accelerator="gpu",
#    devices=1,
#    accumulate_grad_batches=4
#)

#trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
#epoch=25-step=14839.ckpt   epoch=23-step=13167.ckpt epoch=20-step=11286
lv_checkpoint = cwd + '/logs/vit-model/version_8/checkpoints' 
lv_checkpoint2 = cwd + '/logs/vit-model/version_8/checkpoints/epoch=66-step=53504.ckpt' 
checkpoint_callback = ModelCheckpoint(dirpath=lv_checkpoint)#"/content/drive/MyDrive/VisionChallenge/version_2/checkpoints") #version_20/checkpoints")
checkpoint_path = lv_checkpoint2 #"/content/drive/MyDrive/SS3/epoch=19-step=4180.ckpt" #epoch=23-step=5016.ckpt"
earlystop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")
trainer = L.Trainer(callbacks=[checkpoint_callback, earlystop_callback],num_sanity_val_steps=0, max_epochs=100)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=checkpoint_path)

valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
print(valid_metrics)

test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
print(test_metrics)

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
df_epochs = metrics.groupby('epoch').mean()