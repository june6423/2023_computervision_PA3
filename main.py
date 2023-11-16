# import Library
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

# import your model or define here
from torchvision import transforms

# If you want to use args, you can use
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('--gpu', type=str, default='0')
#args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

# This dataset is for coarse segmentation 
# You don't need to use this dataset if you want to make your own dataset
# If you use this dataset, you have to save logits as npz file(corase segmentation)
# It is ok to use data augmentation for this dataset
class SegDataset(torch.utils.data.Dataset): 
  def __init__(self, split='train', seg_path='YOUR_PATH'):

    if split == 'train':
        self.dataset = train_dataset
    elif split == 'val':
        self.dataset = valid_dataset

    self.seg_path = seg_path
    self.split = split

  def __len__(self): 
    return len(self.dataset)
    
  def __getitem__(self, idx): 
    dataset_dic = self.dataset[idx]
    # load npz file and get logits 
    seg = np.load(os.path.join(self.seg_path, f'{self.split}_{self.segment}/{idx}.npz'))['logits'][0]
    seg = torch.from_numpy(seg)
    return dataset_dic, seg

# Download data
root = "."
SimpleOxfordPetDataset.download(root)

train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")

#train_dataset_seg = SegDataset(split='train')
#valid_dataset_seg = SegDataset(split='valid')

print(f"Train size: {len(train_dataset)}")
print(f"val size: {len(valid_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
         # Define all the models you want to use here
         # TODO
        self.model = None
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # This is for coarse segmentation
        #dic, seg_init = batch
        #image = dic['image']
        #input_batch = torch.cat((image, seg_init), dim=1)

        dic = batch
        image = dic['image']
        input_batch = image

        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = dic["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(input_batch)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    model = Model(args)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )