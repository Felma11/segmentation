# ------------------------------------------------------------------------------
# This module contains methods to build datasets.
# ------------------------------------------------------------------------------
#%%
import os
import csv
import random
import sys
import numpy as np
import SimpleITK as sitk
 
from src.data.dataset_obj import Dataset, Instance
from src.utils.load_restore import join_path, pkl_dump, pkl_load
from src.paths import data_paths
from src.utils.introspection import get_class

def restore_if_possible(dataset_name):
    path = os.path.join('storage', 'datasets')
    return pkl_load(path=path, name=dataset_name)

def save_dataset(ds, dataset_name):
    path = os.path.join('storage', 'datasets')
    if not os.path.exists(path):
        os.makedirs(path)
    pkl_dump(ds, path=path, name=dataset_name)

def get_dataset(config):
    name = config['dataset_name']
    # The key gives additional information about the dataset
    dataset_key=config.get('dataset_key', None)
    if dataset_key:
        keyed_name = name + '_' + dataset_key
    else:
        keyed_name = name 
    restore=config.get('restore_dataset', True)
    if restore:
        ds = restore_if_possible(dataset_name=keyed_name)
        if ds:
            print('Restoring existing dataset')
            return ds
    root_path = data_paths[name]
    ds = get_class('src.data.datasets.'+name)(
        root_path=root_path,
        name=keyed_name)
    print('Saving dataset')
    save_dataset(ds, dataset_name=keyed_name)
    return ds

def mnist(root_path,
    name = 'mnist', 
    file_type = 'png', 
    img_shape = (32, 32),
    nr_channels = 1):
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
            instances += [Instance(x=img_paths[i], y=class_dir.name) 
                for i in range(len(img_names))]
            # Define hold-out test set
            if split == 'train':
                hold_out_test_start += len(img_paths)
    hold_out_test_ixs = list(range(hold_out_test_start, len(instances)))
    ds = Dataset(name=name, img_shape=img_shape, file_type=file_type, 
        nr_channels=nr_channels, instances=instances, 
        hold_out_test_ixs=hold_out_test_ixs)
    return ds

def slice_data_to_2D(x, y):
    """
    Converts two arrays of shape (img_size, img_size, nr_slices) 
    into arrays of size (nr_slices, img_size, img_size)
    """
    if (x.shape != y.shape):
        print("Error: Images and Labels do not have the same shape")
    else:
        x = np.array([(x[:, :, z]) for z in range(x.shape[2])])
        y = np.array([(y[:, :, z]) for z in range(y.shape[2])])
    return x, y

key_mapping = {'Institution': 'Institution Name', 'Manufacturer': "Manufacturer's Model Name", 'Protocol': 'Protocol Name'}

patient_categories = {
    'Institution Name': ['Radiologie Fachaerztezentrum', 'Radiologie Nuklearmedizin Adickesallee', 'BILDGEBENDE DIAGNOSTIK', 'Radiologie USZ'],
    "Manufacturer's Model Name": ['Avanto', 'Skyra', 'Ingenia'],
    'Protocol Name': ['t2_tse_tra', 't2_tse_tra_320_p2', 't2_tse_tra_obl']
}

def get_metadata_dict(metadata_path): 
    metadata_dict = dict()
    with open(metadata_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                columns = row
                for col in row:
                    metadata_dict[col] = dict()
            else:
                patient_ix = str(row[0]).rjust(3, '0')
                for col_ix, col_value in enumerate(row):
                    if col_value not in metadata_dict[columns[col_ix]]:
                        metadata_dict[columns[col_ix]][col_value] = []
                    metadata_dict[columns[col_ix]][col_value].append(patient_ix)
            line_count += 1
        return metadata_dict

def patient_names(key, value, metadata_dict):
    return metadata_dict[key][value]

def medcom(root_path,
    name = 'medcom_Manufacturer', 
    file_type = 'mhd', 
    img_shape = None,
    nr_channels = 1):
    # Fetch metadata with key information
    task_key = key_mapping[name.split('_')[1]]
    nr_tasks = len(patient_categories[task_key])
    metadata_dict = get_metadata_dict(os.path.join(root_path, "metadata.csv"))
    patient_task = dict()
    for task_ix, task_value in enumerate(patient_categories[task_key]):
        patient_lst = patient_names(key=task_key, value=task_value, metadata_dict=metadata_dict)
        for patient in patient_lst:
            patient_task[patient] = task_value
    # Fetch images and resample
    #patients = [str(x).rjust(3, '0') for x in range(1, 189)]
    #patients.remove('002')
    instances = []
    # Some are excluded because the number of examples for that key is too small
    for patient in patient_task.keys():
        x = sitk.ReadImage(os.path.join(
            root_path, 'Pat_'+patient+'_img.mhd'))
        y = sitk.ReadImage(os.path.join(
            root_path, 'Pat_'+patient+'_seg.mhd'))
        x_slices = sitk.GetArrayFromImage(x)
        y_slices = sitk.GetArrayFromImage(y)
        assert x_slices.shape == y_slices.shape
        for slice_ix in range(len(x_slices)):
            instances.append(Instance(x=x_slices[slice_ix], y=patient_task[patient],
            mask=y_slices[slice_ix], group_id=patient))
    # Create dataset
    ds = Dataset(name=name, img_shape=img_shape, file_type=file_type, 
        nr_channels=nr_channels, instances=instances, 
        hold_out_test_ixs=[])
    return ds

import sys
def segChallengeProstate(root_path,
    name = 'segChallengeProstate', 
    file_type = 'nii', 
    img_shape = None,
    nr_channels = 1,
    merge_labels=True):
    images_path = os.path.join(root_path, 'imagesTr')
    labels_path = os.path.join(root_path, 'labelsTr')
    filenames = [x for x in os.listdir(images_path) if x[:8] == 'prostate']
    instances = []
    for ix in range(len(filenames)):
        x = sitk.ReadImage(os.path.join(images_path, filenames[ix]))
        x_slices = sitk.GetArrayFromImage(x)[0] # Taking only T2-weighted
        y = sitk.ReadImage(os.path.join(labels_path, filenames[ix]))
        y_slices = sitk.GetArrayFromImage(y)
        assert x_slices.shape == y_slices.shape
        # No longer distinguish between central and peripheral zones
        if merge_labels:
            y_slices = np.where(y_slices==2, 1, y_slices)
        for slice_ix in range(len(x_slices)):
            instances.append(Instance(x=x_slices[slice_ix], y=None,
        mask=y_slices[slice_ix], group_id=filenames[ix]))
    ds = Dataset(name=name, img_shape=img_shape, file_type=file_type, 
        nr_channels=nr_channels, instances=instances, 
        hold_out_test_ixs=[])
    return ds
