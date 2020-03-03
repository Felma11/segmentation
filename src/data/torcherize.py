# ------------------------------------------------------------------------------
# This module transforms a dataset into a PyTorch dataset. Normalizatino values
# can be calculated with src.utils.pytorch.compute_normalization_values.
# ------------------------------------------------------------------------------

from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

transform_pipelines = dict()

norm_transforms = dict()
norm_transforms['segChallengeProstate'] = transforms.Normalize((310.2,), (251.2,))

transform_pipelines['mnist'] = {
    'regular': [transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ],
    'AlexNet': [transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.ToPILImage(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    }

size = (320, 320)
transform_pipelines['medcom'] = {
    'aug': [transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=5, resample=False, fillcolor=0),
        transforms.RandomCrop(size=size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
        transforms.ToTensor()
        ],
    'resize': [transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(size=size),
        transforms.ToTensor()
        ],
    'crop': [transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.CenterCrop(size=size),
        transforms.ToTensor()
        ]
    }

transform_pipelines['segChallengeProstate'] = transform_pipelines['medcom']

class TorchDS(Dataset):
    def __init__(self, dataset_obj, index_list, transform='regular'):
        self.name = dataset_obj.name
        self.set_tranform(transform)

        self.instances = dataset_obj.get_instances(index_list)
        self.filenames = [x.x for x in self.instances]

        self.classes = tuple(sorted(list(dataset_obj.classes)))
        self.nr_classes = len(self.classes)
        self.labels = torch.LongTensor([self.classes.index(x.y) for x in self.instances])

    def set_tranform(self, transform):
        self.transform = transforms.Compose(transform_pipelines[self.name][transform])

    def __len__(self):
        # Return the size of the dataset
        return len(self.filenames)
        
    def __getitem__(self, idx):
        # Open image from path, apply transforms and return with label
        image = Image.open(self.filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]

class TorchSegmentationDataset(Dataset):
    def __init__(self, dataset_obj, index_list, transform='homogenize'):
        self.name = dataset_obj.name.split('_')[0]
        if isinstance(transform, str):
            self.set_tranform(transform)
        else:
            self.transform = transform

        self.instances = dataset_obj.get_instances(index_list)

        self.classes = tuple(sorted(list(dataset_obj.classes)))
        self.nr_classes = len(self.classes)
        self.labels = torch.LongTensor([self.classes.index(x.y) for x in self.instances])

    def set_tranform(self, transform):
        self.transform = transforms.Compose(transform_pipelines[self.name][transform])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        x = self.instances[idx].x.astype(np.int32) #.astype(np.float32)
        y = self.instances[idx].mask.astype(np.int32) #.astype(np.float32)

        # The hack of setting the random seeds is so that image and mask have 
        # the same transformation. 
        # See discussion in https://github.com/pytorch/vision/issues/9        
        seed = np.random.randint(45257265)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(x)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.transform(y)

        #mask = torch.ByteTensor(np.array(mask))

        img = img.float()
        mask = mask.float()

        # Apply normalization only to image
        if self.name in norm_transforms:
            img = norm_transforms[self.name](img)

        return img, mask