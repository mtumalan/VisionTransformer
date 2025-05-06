import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classes import StructuralDamageDataset, StructuralDamageModel
from functions import load_classdict, convertBW

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from sklearn.model_selection import train_test_split

### ----------------------------------------------------------------------------- ###

cwd = os.getcwd()
classdict_path = cwd + '/VisionChallenge/collaboration_it_mx/collaboration_it_mx/output_images/calss_names_colors.csv'
print(classdict_path)

rgb_to_class = load_classdict(classdict_path)

num_classes = len(rgb_to_class)
num_cols = 4
num_rows = (num_classes + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

for i, (rgb, cls) in enumerate(rgb_to_class.items()):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]  # Get the current subplot

    ax.imshow(np.full((10, 10, 3), rgb, dtype=np.uint8))
    ax.set_title(f"Class: {cls}")
    ax.axis('off')  # Turn off axis ticks and labels

for i in range(num_classes, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].axis('off') # Hide subplots not used

bw_dict = convertBW(rgb_to_class)
print(bw_dict)

# Plot the B&W grid
num_classes = len(bw_dict)
num_cols = 4
num_rows = (num_classes + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

for i, (cls, intensity) in enumerate(bw_dict.items()):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]  # Get the current subplot

    # Create a 10x10 patch filled with grayscale intensity
    bw_patch = np.full((10, 10), intensity, dtype=np.uint8)
    ax.imshow(bw_patch, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"Class: {cls}")
    ax.axis('off')  # Turn off axis ticks and labels

# Hide unused subplots
for i in range(num_classes, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].axis('off')

train_path  = cwd + '/VisionChallenge/Attachments/Attachments'
print(train_path)

image_dir = os.path.join(train_path, 'image_png')
mask_dir = os.path.join(train_path, 'mask_png')

image_filenames = os.listdir(image_dir)
mask_filenames = os.listdir(mask_dir)

# Split data: 70% train, 15% validation, 15% test
train_images, temp_images, train_masks, temp_masks = train_test_split(
    image_filenames, mask_filenames, test_size=0.3, random_state=42
)

valid_images, test_images, valid_masks, test_masks = train_test_split(
    temp_images, temp_masks, test_size=0.5, random_state=42
)

# Check split lengths
print(f"Train: {len(train_images)}, Validation: {len(valid_images)}, Test: {len(test_images)}")

# Set datasets with lens
train_dataset = StructuralDamageDataset(image_dir, mask_dir, classdict_path, transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]))
valid_dataset = StructuralDamageDataset(image_dir, mask_dir, classdict_path, transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]))
test_dataset = StructuralDamageDataset(image_dir, mask_dir, classdict_path, transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]))

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, persistent_workers=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, persistent_workers=True)

train_batch = next(iter(train_dataloader))

images, masks = train_batch

print(images.shape)
print(masks.shape)

print(train_batch[0][0])
print(train_batch[1][0])

image = train_batch[0][1].permute(1, 2, 0).numpy()
mask = train_batch[1][1].type(torch.int64).numpy()

print(image.shape)
print(mask.shape)

# Load the class-to-color map
classdict = pd.read_csv(classdict_path)
color_map = {row['name']: [row['r'], row['g'], row['b']] for _, row in classdict.iterrows()}  # Map class names to RGB colors

print("Loaded color map: ", color_map)

# Create a mapping from class names to indices
class_to_index = {class_name: index for index, class_name in enumerate(color_map.keys())}

print("Class to index mapping: ", class_to_index)

# --- Model ---

model = StructuralDamageModel("unet", "resnet34", in_channels=3, out_classes=17)

# Early stop is a callback that is used to stop the training process when the validation loss does not improve. In this case, we are
# using the EarlyStopping callback to stop the training process when the validation loss does not improve for 3 epochs.
earlystop_callback = EarlyStopping(
    monitor="valid_loss",  # Metric to monitor
    patience=3,            # Number of epochs to wait for improvement
    verbose=True,          # Print logs about early stopping
    mode="min"             # Minimize the monitored metric
)

# Define a logger to save logs
logger = CSVLogger(save_dir="logs/", name="struct-model")

# Define the Trainer with the callbacks and logger
trainer = L.Trainer(
    max_epochs=10,               # Maximum number of epochs
    logger=logger,               # Save logs in a CSV file
    callbacks=[earlystop_callback],  # Add EarlyStopping to callbacks
    accelerator="gpu",           # Use GPU if available
    devices=1,                   # Number of GPUs to use (set to 1 here)
    accumulate_grad_batches=4    # Accumulate gradients over 2 batches
)

# Start training
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

# --- Cell Separator ---

valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
print(valid_metrics)

# --- Cell Separator ---

test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
print(test_metrics)

# Read the metrics.csv file generated by the PyTorch Lightning logger
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

# Group the metrics by epoch and compute the mean loss for each epoch
df_epochs = metrics.groupby('epoch').mean()

#display(df_epochs)