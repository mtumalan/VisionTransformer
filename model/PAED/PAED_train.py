#Azad_PAED_Loss_Training

import os
import torch
import cv2
import time
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torchmetrics
from torch.nn import CrossEntropyLoss
from torchvision.transforms import functional as FF

cwd = os.getcwd()
train_path  = cwd + '/../../VisionChallenge/Attachments/Attachments'
print(train_path)

IMAGE_DATASET_PATH = os.path.join(train_path, 'image_png')
TRAIN_MASK_DATASET_PATH = os.path.join(train_path, 'mask_png')


#IMAGE_DATASET_PATH = "/home/droni/Documents/azade/UnetAzadeCrackDetect_original/crack_segmentation_dataset/img_png"
#TRAIN_MASK_DATASET_PATH = "/home/droni/Documents/azade/UnetAzadeCrackDetect_original/crack_segmentation_dataset/mask_png"
VAL_MASK_DATASET_PATH = TRAIN_MASK_DATASET_PATH
TEST_SPLIT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

loss_name='paed'
num_images=4000
NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 3

INIT_LR = 0.001
NUM_EPOCHS = 15
BATCH_SIZE = 32

INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

THRESHOLD = 0.35

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, f"{loss_name}_loss.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, f"{loss_name}_plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, f"{loss_name}_test_paths.txt"])

MODEL_PATH=os.path.join(BASE_OUTPUT,f"{loss_name}_loss.pth")

#Dataset
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	def __len__(self):
		return len(self.imagePaths)
	
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx],cv2.IMREAD_GRAYSCALE)
		mask=(mask>0).astype(np.float32)
		
		if self.transforms is not None:
			image = self.transforms(image)
			# mask = torch.tensor(mask, dtype=torch.float32)
			# mask = FF.to_pil_image(mask)
			mask = self.transforms(mask)			
		return (image, mask)
	


#Unet
class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)

	def forward(self, x):
		return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		self.encBlocks = ModuleList([Block(channels[i], channels[i + 1])	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	
	def forward(self, x):
		blockOutputs = []
		for block in self.encBlocks:
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		return blockOutputs

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		self.channels = channels
		self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2)	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList([Block(channels[i], channels[i + 1])	for i in range(len(channels) - 1)])
	
	def forward(self, x, encFeatures):
		for i in range(len(self.channels) - 1):
			x = self.upconvs[i](x)
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		return x
	
	def crop(self, encFeatures, x):
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		return encFeatures

class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=NUM_CLASSES, retainDim=True, outSize=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
		super().__init__()
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize

	def forward(self, x):
		encFeatures = self.encoder(x)
		decFeatures = self.decoder(encFeatures[::-1][0],encFeatures[::-1][1:])
		map = self.head(decFeatures)
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		return map
        

def paed_loss(msk, pred_mask, epoch,threshold=0.35):
    
    batch_size = msk.size(0)
    total_paed = torch.zeros(1, device=msk.device,requires_grad=True)
    start_time = time.time()  # Start measuring time
    
    for b in range(batch_size):
        print(f"Epoch {epoch + 1}, Processing batch {b + 1}/{batch_size}")
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
            paed = torch.zeros(1,device=msk.device)

            
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
                distances = torch.sqrt(torch.sum((mask_coordinates[i] - prediction_coordinates)**2, dim=1))
                distance_x1i_S2[i] = torch.min(distances)
            
            distance_S1_S2 = torch.sum(distance_x1i_S2)
            
           
            distance_x2j_S1 = torch.zeros(m, device=msk.device)
            for j in range(m):
                distances = torch.sqrt(torch.sum((prediction_coordinates[j] - mask_coordinates)**2, dim=1))
                distance_x2j_S1[j] = torch.min(distances)
            
            distance_S2_S1 = torch.sum(distance_x2j_S1)
            
            paed = (distance_S1_S2 + distance_S2_S1+0.001) / (n + m+0.001)
        
        total_paed =total_paed + paed
        
    end_time = time.time()  # End measuring time
    print(f"paed_loss execution time: {end_time - start_time:.2f} seconds")
    return total_paed / batch_size               




#training

imagePaths = sorted([
    os.path.join(IMAGE_DATASET_PATH, f)
    for f in os.listdir(IMAGE_DATASET_PATH)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
])
imagePaths=imagePaths[:num_images]

trainMaskPaths = [os.path.join(TRAIN_MASK_DATASET_PATH, os.path.basename(p)) for p in imagePaths]
valMaskPaths = [os.path.join(VAL_MASK_DATASET_PATH, os.path.basename(p)) for p in imagePaths]

trainImages, valImages = train_test_split(imagePaths, test_size=TEST_SPLIT, random_state=42)
trainMasks, valMasks = train_test_split(trainMaskPaths, test_size=TEST_SPLIT, random_state=42)

print("[INFO] saving testing image paths...")
f = open(TEST_PATHS, "w")
f.write("\n".join(valMasks))
f.close()

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT,	INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])  

trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,	transforms=transforms)
valDS = SegmentationDataset(imagePaths=valImages, maskPaths=valMasks,transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(valDS)} examples in the test set...")


trainLoader = DataLoader(trainDS, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,num_workers=os.cpu_count())
valLoader = DataLoader(valDS, shuffle=False,batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,num_workers=os.cpu_count())


unet = UNet().to(DEVICE)

opt = Adam(unet.parameters(), lr=INIT_LR)

calculate_accuracy = torchmetrics.Accuracy(task='binary', threshold=THRESHOLD).to(DEVICE) 
calculate_iou = torchmetrics.classification.BinaryJaccardIndex(threshold=THRESHOLD).to(DEVICE)  


trainSteps = len(trainDS) // BATCH_SIZE
valSteps = len(valDS) // BATCH_SIZE

H = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_iou": [], "val_iou": []}



print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
	unet.train()
	totalTrainLoss = 0
	totalValLoss = 0
	totalTrainAcc = 0
	totalValAcc = 0
	totalTrainIoU = 0
	totalValIoU = 0


	for (i, (x, y)) in enumerate(trainLoader):
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
		y=(y>0.35).float()
		pred = unet(x)
		pred = torch.sigmoid(pred) 

		if i == 0 and e == 0:  
					print(f"Mask shape: {y.shape}, type: {y.dtype}, min: {y.min().item()}, max: {y.max().item()}")
					print(f"Image shape: {x.shape}, type: {x.dtype}, min: {x.min().item()}, max: {x.max().item()}")
					print(f"Prediction shape: {pred.shape}, type: {pred.dtype}, min: {pred.min().item()}, max: {pred.max().item()}")



		loss_t = paed_loss(y, pred, e)
		opt.zero_grad()
		loss_t.backward()
		opt.step()
		totalTrainLoss += loss_t.item()
		pred_binary = (torch.sigmoid(pred) > THRESHOLD).float()
		
		if i == 0 and e == 0: 
					print(f"Mask shape: {y.shape}, type: {y.dtype}, min: {y.min().item()}, max: {y.max().item()}")
					print(f"Image shape: {x.shape}, type: {x.dtype}, min: {x.min().item()}, max: {x.max().item()}")
					print(f"Prediction shape: {pred_binary.shape}, type: {pred_binary.dtype}, min: {pred_binary.min().item()}, max: {pred_binary.max().item()}")

		totalTrainAcc += calculate_accuracy(pred_binary, y).item()
		totalTrainIoU += calculate_iou(pred_binary, y).item()

	with torch.no_grad():
		unet.eval()
		for (x, y) in valLoader:
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			y=(y>0.35).float()
			pred_v0 = unet(x)
			pred_v0 = torch.sigmoid(pred_v0) 
			totalValLoss +=paed_loss(y,pred_v0, e).item()
			pred_binary = (torch.sigmoid(pred_v0) > THRESHOLD).float()

			totalValAcc += calculate_accuracy(pred_binary, y).item()
			totalValIoU += calculate_iou(pred_binary, y).item()

	avgTrainLoss = totalTrainLoss / len(trainLoader)
	avgValLoss = totalValLoss / len(valLoader)
	avgTrainAcc = totalTrainAcc / len(trainLoader)
	avgValAcc = totalValAcc / len(valLoader)
	avgTrainIoU = totalTrainIoU / len(trainLoader)
	avgValIoU = totalValIoU / len(valLoader)

	H["train_loss"].append(avgTrainLoss)
	H["val_loss"].append(avgValLoss)
	H["train_acc"].append(avgTrainAcc)
	H["val_acc"].append(avgValAcc)
	H["train_iou"].append(avgTrainIoU)
	H["val_iou"].append(avgValIoU)

	print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
	print(f"Train loss: {avgTrainLoss:.6f},   Train acc: {avgTrainAcc:.4f},   Train IoU: {avgTrainIoU:.4f},  Val loss: {avgValLoss:.4f},   Val acc: {avgValAcc:.4f},   Val IoU: {avgValIoU:.4f}")

	

endTime = time.time()

print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")
class plot(): 

	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H["train_loss"], label="train_loss")
	plt.plot(H["val_loss"], label="val_loss")
	plt.title("Training and Validation Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(os.path.join(BASE_OUTPUT, f"{loss_name}_loss_plot.png"))

	
	plt.figure()
	plt.plot(H["train_acc"], label="train_acc")
	plt.plot(H["val_acc"], label="val_acc")
	plt.title("Training and Validation Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(os.path.join(BASE_OUTPUT, f"{loss_name}_accuracy_plot.png"))

	
	plt.figure()
	plt.plot(H["train_iou"], label="train_iou")
	plt.plot(H["val_iou"], label="val_iou")
	plt.title("Training and Validation IoU")
	plt.xlabel("Epoch #")
	plt.ylabel("IoU")
	plt.legend(loc="lower left")
	plt.savefig(os.path.join(BASE_OUTPUT, f"{loss_name}_iou_plot.png"))


torch.save(unet.state_dict(), MODEL_PATH)