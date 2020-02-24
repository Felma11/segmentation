# ------------------------------------------------------------------------------
# This module contains methods to build datasets.
# ------------------------------------------------------------------------------

import os
import random
import sys
 
from src.data.dataset_obj import Dataset, Instance
from src.utils.load_restore import pkl_dump, pkl_load

def restore_if_possible(dataset_name):
    path = os.path.join('storage', 'datasets')
    return pkl_load(path=path, name=dataset_name)

def save_dataset(ds, dataset_name):
    path = os.path.join('storage', 'datasets')
    pkl_dump(ds, path=path, name=dataset_name)

def mnist(root_path,
    name = 'MNIST', 
    file_type = 'png', 
    img_shape = (32, 32),
    nr_channels = 1,
    restore = True,
    val_ratio = 0.0,
    stratesfied = True):
    if restore:
        ds = restore_if_possible(name)
        if ds:
            return ds
    # The root includs directories 'train' and 'test'
    instances = []
    hold_out_test_start = 0
    for split in ['train', 'test']:
        path = os.path.join(root_path, split)
        # Each includes a directory for the instances of each class, with the 
        # name of the class as directory name
        for class_dir in os.scandir(path):    
            img_names = sorted(img.name for img in os.scandir(class_dir))
            img_paths = [os.path.join(class_dir.path, img) for img in img_names]
            instances += [Instance(x_path=img_paths[i], y=class_dir.name) 
                for i in range(len(img_names))]
            # Define hold-out test set
            if split == 'train':
                hold_out_test_start += len(img_paths)
    hold_out_test_ixs = list(range(hold_out_test_start, len(instances)))
    ds = Dataset(name=name, img_shape=img_shape, file_type=file_type, 
        nr_channels=nr_channels, instances=instances, 
        hold_out_test_ixs=hold_out_test_ixs)
    return ds