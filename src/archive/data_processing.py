import os
import pickle
import numpy as np
import SimpleITK as sitk

from src.utils.load_restore import join_path

def load_data(data_dir, resample_size, resample_spacing):
    """
    Loads .mhd files and converts them into numpy arrays of size
    (nr_examples, img_size, img_size, nr_slices)
    """
    image_filenames, label_filenames = zip(*list(iterate_folder(data_dir)))
    x = np.array([resample_img(sitk.ReadImage(f), "Image", resample_size, resample_spacing) for f in image_filenames])  
    y = np.array([resample_img(sitk.ReadImage(f), "Label", resample_size, resample_spacing) for f in label_filenames])     
    return x,y

#Assumes that all images end with img.mhd and segmentations with seg.mhd
def iterate_folder(folder):
    for filename in sorted(os.listdir(folder)):
        if(filename[-7:] == "img.mhd"):
            img_filename = join_path([folder, filename])
            seg_filename = img_filename[:-7] + 'seg.mhd'
            if os.path.exists(seg_filename):
                yield img_filename, seg_filename

def resample_img(img, img_type, size, spacing):
    """
    Resample 3D images to a standard size and spacing.
    :returns: two arrays of shape (len(img), size, size, spacing)
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    if img_type is "Label":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif img_type is "Image":
        resampler.SetInterpolator(sitk.sitkLinear)
    imgResampled = resampler.Execute(img)

    #axis have to be switched since np.array and keras use them in different order...
    x = np.transpose(sitk.GetArrayFromImage(imgResampled).astype(dtype=np.float), [2, 1, 0])
    return x

def split_data(x, y, split_percent):
    split_index = np.arange(len(x))
    np.random.shuffle(split_index)
    split_num = int(x.shape[0]*split_percent)
    x_train = x[split_index[split_num:]]
    x_val = x[split_index[:split_num]]
    y_train = y[split_index[split_num:]]
    y_val = y[split_index[:split_num]]
    return x_train, x_val, y_train, y_val

def slice_data_to_2D(x, y):
    """
    Converts two arrays of shape (nr_examples, img_size, img_size, nr_slices) 
    into arrays of size (nr_examples*nr_slices, img_size, img_size)
    """
    if(x.shape != y.shape):
        print("Error: Images and Labels do not have the same shape")
    else:
        x = np.array([(x[i, :, :, z]) for i in range(x.shape[0]) for z in range(x.shape[3])])
        y = np.array([(y[i, :, :, z]) for i in range(y.shape[0]) for z in range(y.shape[3])])
    return x,y