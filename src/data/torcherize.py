# ------------------------------------------------------------------------------
# This module transforms a dataset into a PyTorch dataset. Normalizatino values
# can be calculated with src.utils.pytorch.compute_normalization_values.
# ------------------------------------------------------------------------------

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

transform_pipelines = dict()

transform_pipelines['mnist'] = {'regular': [transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.Pad(2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]}

class TorchDS(Dataset):
    def __init__(self, dataset_obj, index_list, transform='regular', classification=True):
        self.transform = transforms.Compose(transform_pipelines[dataset_obj.name][transform])
        self.classes = tuple(sorted(list(dataset_obj.classes)))
        self.nr_classes = len(self.classes)
        self.instances = dataset_obj.get_instances(index_list)
        if classification:
            self.labels = torch.LongTensor([self.classes.index(x.y) for x in self.instances])
        self.filenames = [x.x_path for x in self.instances]
        
    def __len__(self):
        # Return the size of the dataset
        return len(self.filenames)
        
    def __getitem__(self, idx):
        # Open image from path, apply transforms and return with label
        image = Image.open(self.filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]