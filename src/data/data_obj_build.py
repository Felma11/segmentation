# ------------------------------------------------------------------------------
# This module contains methods to build datasets.
# ------------------------------------------------------------------------------

import os
import random
import sys
 
from cl.data.data_obj import Dataset, Instance
from cl.utils.load_restore import pkl_dump, pkl_load

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


            
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    directory_splits = ['train', 'test']
    instances = {'train':[], 'val':[], 'test':[]}
    # The root inclused directories 'train' and 'test'
    for split in directory_splits:
        path = os.path.join(root_path, split)
        # Each includes a directory for the instances of each class, with 
        # the name of the class as directory name
        for class_dir in os.scandir(path):    
            img_names = sorted(img.name for img in os.scandir(class_dir))
            img_paths = [os.path.join(class_dir.path, img) for img in img_names]
            img_instances = [Instance(img_names[i], img_paths[i], class_dir.name) 
                for i in range(len(img_names))]
            instances[split].extend(img_instances)
        shuffle(instances[split])
    if not stratesfied:
        # Randomly select the percentage of maintained examples from 100% to 10%
        instances = unstratesfy(instances, classes)
    if val_ratio > 0:
        instances['train'], instances['val'], _ = shuffle_divide(instances['train'], classes, vali_rate=val_ratio, test_rate=0)
    ds = Dataset(name, classes, file_type, img_shape, nr_channels, 
        instances['train'], instances['val'], instances['test'])
    save_dataset(ds, name)
    return ds

    Instance(x_path, y=None)


    Dataset(name=name, file_type=file_type, img_shape=img_shape, nr_channels=nr_channels, classes=classes, instances = [], test_ixs = [])