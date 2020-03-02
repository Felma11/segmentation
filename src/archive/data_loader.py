import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from src.utils.load_restore import pkl_load
import numpy as np
from PIL import Image

from src.archive.data_processing import split_data, slice_data_to_2D
 
class MyDataset(Dataset):
    def __init__(self, imgs, masks, transform_image=None, transform_mask=None):
        self.input_images, self.target_masks = imgs, masks
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform_image:
            image = self.transform_image(image)
    
        if self.transform_mask:
            mask = self.transform_mask(mask)
          
        #Model needs float tensors; ToTesor() in transform should return float tensor but doesnt -.-
        image = image.float()
        mask = mask.float()
        return [image, mask]

def setup_dataloaders(storage_dir, batch_size):
    x, y = pkl_load("data_dump", storage_dir)
    #x = x.astype("uint8")
    #y = y.astype("uint8")
    x_train, x_val, y_train, y_val = split_data(x, y, split_percent = 0.2)
    x_train, y_train = slice_data_to_2D( x_train, y_train)
    x_val, y_val = slice_data_to_2D(x_val, y_val)

    #TODO: random deform, brightness stuff, param selection, check if normalize works
    #TODO: same augmentation has to be applied on mask, e.g. use seed
    trans_img = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomRotation(degrees=10, fill=(0,)), 
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        #transforms.Normalize([np.mean(x)], [np.std(x)])
    ])
    trans_mask = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    train_set = MyDataset(x_train, y_train, trans_img, trans_mask)
    val_set = MyDataset(x_val, y_val, trans_img, trans_mask)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders