# ------------------------------------------------------------------------------
# PyTorch requires the mean and standard deviation to be calculated manually for
# normalization. This method can be used for that.
# ------------------------------------------------------------------------------

#%%
from IPython import get_ipython
get_ipython().magic('load_ext autoreload') 
get_ipython().magic('autoreload 2')

import torch
from torchvision import transforms

from src.data.datasets import get_dataset
from src.data.torcherize import TorchSegmentationDataset

def normalization_values(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    count = 0
    mean = torch.empty(3)
    std = torch.empty(3)

    for data, _ in dataloader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        mean = (count * mean + sum_) / (count + nb_pixels)
        std = (count * std + sum_of_square) / (count + nb_pixels)
        count += nb_pixels

    return mean, torch.sqrt(std - mean ** 2)

#%%
# segChallengeProstate
dataset = get_dataset({"dataset_name": "segChallengeProstate"})
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize(size=(320, 320)),
    transforms.ToTensor()])
pytorch_dataset = TorchSegmentationDataset(dataset_obj=dataset, 
    index_list=list(range(len(dataset.instances))), transform=transform)
print(normalization_values(pytorch_dataset))

# %%
