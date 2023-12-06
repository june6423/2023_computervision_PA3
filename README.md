# 2023_computervision_PA3
Before start
1. pip3 install torch torchvision torchaudio
2. pip install Pillow~=9.5
3. pip install segmentation_models_pytorch
4. Replace /anaconda3/envs/pa3/lib/python3.8/site-packages/segmentation_models_pytorch/datasets/oxford_pet.py into given oxford_pet.py (Ctrl + left click on SimpleOxfordPetDataset in coarse.py)
    Only line 39 in oxford_pet.py is modified

How to run
1. python coarse.py #U-Net segmentation code
2. python refine.py #CSPN refinement code

!!Please check the version!!
python = 3.8
pytorch = 2.1.0
Pillow = 9.5
CUDA = 12.2
