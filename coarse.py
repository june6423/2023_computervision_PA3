import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from model import UNet,get_iou
import matplotlib.pyplot as plt

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

root = "."
#root = "." + "/CV2023_PA3"
maskpath = root + "/mask/"
#SimpleOxfordPetDataset.download(root)

n_cpu = os.cpu_count()
train_dataset = SimpleOxfordPetDataset(root=root, mode="train")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)

val_dataset = SimpleOxfordPetDataset(root=root, mode="valid")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)

model = UNet(n_channels=3,n_classes=1)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss() 

print("Everything is ready! Device is :", device)
# Training loop
log_interval = 100 
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        image = sample['image'] / 255
        mask = sample['mask'] 
        image = image.to(device).float()
        mask = mask.to(device).float()
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
            
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_IoU = 0.0
    with torch.no_grad():
        for sample in val_loader:
            image = sample['image'] /255
            mask = sample['mask']
            
            image = image.to(device).float()
            mask = mask.to(device).float()
            output = model(image)
            val_loss += criterion(output, mask)
            val_IoU += get_iou(output, mask)

    val_loss /= len(val_loader)
    val_IoU /= len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss} Validation IoU: {val_IoU}")


# Switch model to evaluation mode
model.eval()


predicted_masks = []
os.makedirs(maskpath, exist_ok=True)

for batch_idx, sample in enumerate(train_loader):
    image = sample['image'] / 255
    image = image.to(device).float()
    filename = sample['filename']
    
    with torch.no_grad():
        output = model(image)

    binary_masks = (output> 0.5).float() 
    for mask_idx, (binary_mask, filename) in enumerate(zip(binary_masks, filename)):
        # Convert tensor to numpy array
        binary_mask_numpy = binary_mask.cpu().numpy()
        binary_mask_numpy = binary_mask_numpy.squeeze(0)

        # Save the mask as an image using matplotlib 
        plt.imshow(binary_mask_numpy) 
        plt.axis('off')
        plt.savefig(os.path.join(maskpath, f'{os.path.basename(filename)}.png'),bbox_inches='tight', pad_inches=0)
        plt.close()

for batch_idx, sample in enumerate(val_loader):
    image = sample['image'] / 255
    image = image.to(device).float()
    filename = sample['filename']
    
    with torch.no_grad():
        output = model(image)

    binary_masks = (output> 0.5).float() 
    for mask_idx, (binary_mask, filename) in enumerate(zip(binary_masks, filename)):
        # Convert tensor to numpy array
        binary_mask_numpy = binary_mask.cpu().numpy()
        binary_mask_numpy = binary_mask_numpy.squeeze(0)

        # Save the mask as an image using matplotlib 
        plt.imshow(binary_mask_numpy) 
        plt.axis('off')
        plt.savefig(os.path.join(maskpath, f'{os.path.basename(filename)}.png'),bbox_inches='tight', pad_inches=0)
        plt.close()
