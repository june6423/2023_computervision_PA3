import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SegDataset(Dataset):
    def __init__(self,root, root_rgb, root_masks, mode):
        self.root = root
        self.root_rgb = root_rgb
        self.root_masks = root_masks
        self.mode = mode

        self.roots_gt = os.path.join(self.root, "annotations", "trimaps")
        self.filenames = self._read_split()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.root_rgb, filename + ".jpg")
        mask_path = os.path.join(self.root_masks, filename + ".png")
        gt_path = os.path.join(self.roots_gt, filename + ".png")
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        trimap = np.array(Image.open(gt_path))
        gt = self._preprocess_mask(trimap)

        image = np.array(Image.fromarray(image).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
        gt = np.array(Image.fromarray(gt).resize((256, 256), Image.NEAREST))

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 30
        gt = gt.astype(np.float32)
        # convert to other format HWC -> CHW
        
        sample = dict(image=image, mask=mask,filename=filename,gt=gt)
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["gt"] = np.expand_dims(gt, 0)
        return sample
    
    def _read_split(self):
        split_filename = "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames
    
    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask