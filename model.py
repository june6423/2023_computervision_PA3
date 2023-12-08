import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self,x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self,x1, x2):
        x1 = self.up(x1)        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x) 
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels,n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.classes = n_classes
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64,n_classes))
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_with_attention(nn.Module):
    def __init__(self, n_channels,n_classes,n_iter, bilinear=False):
        super(UNet_with_attention, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.classes = n_classes
        self.n_iter = n_iter
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64,n_classes))
        self.attention = (DoubleConv(64,4*n_iter))
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        attention = self.sigmoid(self.attention(x))
        reshaped = attention.view(attention.size(0), 4, -1, attention.size(2), attention.size(3)) # b, 4, n_iter, h, w
        return logits, reshaped

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.attention = torch.utils.checkpoint(self.attention)

class CSPN(nn.Module):
    def __init__(self):
        super(CSPN, self).__init__()
	# affinity: b, 9, h, w (sum to 1 in dimension 1)
	# current_segmentation: b, 1, h, w
	# coarse_segmentation: b, 1, h, w

    def forward(self,affinity, current_segmentation, coarse_segmentation):	
        assert affinity.shape[1] == 8
        
        abs_sum = torch.sum(torch.abs(affinity), dim=1, keepdim=True)
        sum = torch.sum(affinity, dim=1, keepdim=True)
        affinity = affinity / abs_sum
        new_affinity = torch.cat((affinity[:,0:4,:,:], 1-sum,affinity[:,5:,:,:]), dim=1)
         
        assert new_affinity.shape[1] == 9
        new_affinity = new_affinity.view(new_affinity.size(0),9,-1) #Now sum of new_affinity is 1!
        
        unfold = nn.Unfold(kernel_size=3, stride=1,padding=1)
        fold = nn.Fold(output_size=(256,256),kernel_size=3, stride=1,padding=1)
        current_segmentation_unfold = unfold(current_segmentation) #b, 9, h*w
        coarse_segmentation_unfold = unfold(coarse_segmentation)[:,4,:].unsqueeze(1) #b, 1, h*w
        unfolded_seg = torch.cat((current_segmentation_unfold[:,0:4,:], coarse_segmentation_unfold), dim=1) 
        unfolded_seg = torch.cat((unfolded_seg, current_segmentation_unfold[:,5:,:]), dim=1) #b, 9, h*w
        
        output = fold(new_affinity * unfolded_seg)
        return output # b, 1, h, w

class DYSPN(nn.Module):
    def __init__(self):
        super(DYSPN, self).__init__()
        self.index = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        assert len(self.index) == 49
    # index is the index of the 7*7 affinity matrix
    
    # 0 0 0 0 0 0 0
    # 0 1 1 1 1 1 0 
    # 0 1 2 2 2 1 0
    # 0 1 2 3 2 1 0
    # 0 1 2 2 2 1 0
    # 0 1 1 1 1 1 0
    # 0 0 0 0 0 0 0
    
	# affinity: b, 49, h, w (sum to 1 in dimension 1)
	# current_segmentation: b, 1, h, w
	# coarse_segmentation: b, 1, h, w

    def forward(self,affinity, attention,current_segmentation, coarse_segmentation):	
        assert affinity.shape[1] == 49 # b,49,h,w
        assert attention.shape[1] == 4 # b,4,h,w
        assert current_segmentation.shape[1] == 1 # b,1,h,w
        assert coarse_segmentation.shape[1] == 1 # b,1,h,w
        
        unfold = nn.Unfold(kernel_size=7, stride=1,padding=3)
        fold = nn.Fold(output_size=(256,256),kernel_size=7, stride=1,padding=3)
        
        current_segmentation_unfold = unfold(current_segmentation)
    
        S_ppt = self.attention_conv(affinity, attention) + attention[:,3,:,:] #B, H, W
        S_prime_ppt = self.attention_conv(torch.abs(affinity), attention) + attention[:,3,:,:]
        
        kernel = torch.zeros(attention.shape[0],49,attention.shape[2],attention.shape[3]).to(attention.device) #B, 49, H, W
        for i in range(49):
            if(self.index[i] != 3):
                kernel[:,i,:,:] = attention[:,self.index[i],:,:] #B, 49, H, W
        kernel = kernel * affinity
        kernel = kernel.view(kernel.size(0),49,-1) #B, 49, H*W
        
        output = fold(kernel * current_segmentation_unfold).squeeze(1)
        output = output / (S_prime_ppt + 1e-6)
        output = output + attention[:,3,:,:]/(S_prime_ppt + 1e-6) * (current_segmentation.squeeze(1))
        output = output + (1-S_ppt/(S_prime_ppt + 1e-6)) * (coarse_segmentation.squeeze(1)) #Now sum to 1
        
        return output.unsqueeze(1) # b, 1, h, w

    def attention_conv(self, affinity, attention):
        assert affinity.shape[1] == 49 # b, 49, h, w
        assert attention.shape[1] == 4 # b, 4, h, w
        
        new_attention = torch.zeros(attention.shape[0],49,attention.shape[2],attention.shape[3]).to(attention.device) #B, 49, H, W
        for i in range(49):
            if(self.index[i] != 3):
                new_attention[:,i,:,:] = attention[:,self.index[i],:,:] #B, 49, H, W
        
        attention_sum = torch.sum(new_attention*affinity,axis=1)
        return attention_sum #S_ij와 \hat(S)_ij 모두 사용할 수 있는 함수
        
def get_iou(prediction, target):
    # Convert images to numpy arrays
    prediction_np = np.array(prediction.cpu())
    target_np = np.array(target.cpu())
    prediction_np[prediction_np<0] = 0
    prediction_np[prediction_np>0] = 1
    prediction_np = prediction_np.astype(np.int16)
    target_np = target_np.astype(np.int16)
    # Calculate intersection and union
    intersection = np.logical_and(prediction_np, target_np).sum()
    union = np.logical_or(prediction_np, target_np).sum()
    
    #print("intersection", intersection)
    #print("union", union)
    # Calculate IoU
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou