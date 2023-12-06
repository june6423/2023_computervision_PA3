import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNet_with_attention, DYSPN, get_iou
import matplotlib.pyplot as plt
import numpy as np
from dataloader import SegDataset
import time

root = "."
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#root = "." + "/CV2023_PA3"
maskpath = root + "/mask/"
imagepath = root + "/images/"
dyspnpath = root + "/dyspn/"

n_cpu = os.cpu_count()
train_dataset = SegDataset(root = root, root_rgb=imagepath, root_masks=maskpath,mode="train")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

val_dataset = SegDataset(root = root, root_rgb=imagepath, root_masks=maskpath, mode="valid")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)


unet = UNet_with_attention(n_channels=4,n_classes=49,n_iter=6)
dyspn = DYSPN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet.to(device)
dyspn.to(device)
optimizer = torch.optim.Adam(list(unet.parameters()) + list(dyspn.parameters()), lr=0.001)

criterion = nn.BCEWithLogitsLoss()  

print("Everything is ready! Device is :", device)
# Training loop
log_interval = 100 
num_epochs = 40
num_dyspn = 6

unet.train()
dyspn.train()

os.makedirs(dyspnpath, exist_ok=True)

for epoch in range(num_epochs):
    start = time.time()
    for batch_idx, sample in enumerate(train_loader):
        image = sample['image']
        mask = sample['mask']
        gt = sample['gt']
        
        image = image.to(device).float()
        mask = mask.to(device).float()
        gt = gt.to(device).float()
        
        optimizer.zero_grad()
        input = torch.cat((image, mask), dim=1)
        affinity,attention = unet(input) #affinity B,49,H,W attention B,4,num_dyspn,H,W
        current_segmentation = mask
        
        assert attention.shape[2] == num_dyspn
        for i in range(num_dyspn):
            current_segmentation = dyspn(affinity,attention[:,:,i,:,:],current_segmentation, mask)
        loss = criterion(current_segmentation,gt)
        loss.backward()
        optimizer.step()
    # Validation loop
    unet.eval()
    val_loss = 0.0
    val_IoU = 0.0
    
    with torch.no_grad():
        for sample in val_loader:
            image = sample['image'] 
            mask = sample['mask']
            gt = sample['gt']
            filename = sample['filename'][0]
            
            image = image.to(device).float()
            mask = mask.to(device).float()
            gt = gt.to(device).float()
            
            input = torch.cat((image, mask), dim=1)
            affinity,attention = unet(input)
            current_segmentation = mask
            
            assert attention.shape[2] == num_dyspn
            
            for i in range(num_dyspn):
                current_segmentation = dyspn(affinity, attention[:,:,i,:,:],current_segmentation, mask)
                if(filename == 'Abyssinian_100'):
                    binary_masks = (current_segmentation > 0).float() 
                    binary_mask_numpy = binary_masks.cpu().numpy()
                    binary_mask_numpy = binary_mask_numpy.squeeze(0)
                    binary_mask_numpy = binary_mask_numpy.squeeze(0)
                    
                    # Save the mask as an image using matplotlib 
                    plt.imshow(binary_mask_numpy) 
                    plt.axis('off')
                    figurename = filename +'_Epoch_'+str(epoch)+'_Iter_'+str(i)
                    plt.savefig(os.path.join(dyspnpath, f'{os.path.basename(figurename)}.png'),bbox_inches='tight', pad_inches=0)
                    plt.close()
                    for j in range(4):
                        plt.imshow(attention[0,j,i,:,:].cpu().numpy()) 
                        plt.axis('off')
                        figurename = filename +'_Epoch_'+str(epoch)+'_Iter_'+str(i)+'_Attention_'+str(j)
                        plt.savefig(os.path.join(dyspnpath, f'{os.path.basename(figurename)}.png'),bbox_inches='tight', pad_inches=0)
                        plt.close()
            val_loss += criterion(current_segmentation, gt)
            val_IoU += get_iou(current_segmentation, gt)
            if(torch.isnan(val_loss)):
                print("NAN")
                break
    val_loss /= len(val_loader)
    val_IoU /= len(val_loader)
    end = time.time()
    print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss} Validation IoU: {val_IoU} Time: {end-start}")
