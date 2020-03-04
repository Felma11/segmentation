# ------------------------------------------------------------------------------
# The evaluation consists of cropping images in a sliding-window way and passing
# them through the network. 
# ------------------------------------------------------------------------------
  
# A dataset is built with all cropped images. Using the index range which
# make up an image, as well as the dictionary stating when a patch starts and 
# when one ends

# For regions which take up the same space, select through voting (is the
# average greater than 0.5) whither the classification is a 0 or a 1.

#torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
#torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

# Split image into patches, determined through patch size and stride

# Pass each patch through the model

# Threshold the segmentation to secide on the label
#%%
import numpy as np
from src.eval.patch_based_eval.patch_logic import generate_patches, merge_crops_overlapping, merge_crops_averaging, threshold_mask
img = np.zeros((3, 4))
mask = np.zeros((3, 4))
shape = img.shape
patches = generate_patches(img=mask, patch_side=2, stride=1)
img_crops = [patch.crop(img=img) for patch in patches]
mask_crops = [patch.crop(img=mask) for patch in patches]
assert len(patches) == len(img_crops) == len(mask_crops) == 6
#%%
img_crops[0].fill(2)
merged_img = merge_crops_overlapping(patches, crops=img_crops, shape=shape)
assert np.allclose(merged_img, np.array([[2., 2., 0., 0.], [2., 2., 0., 0.], [0., 0., 0., 0.]]))
#%%
mask_crops[0]=4
merged_mask = merge_crops_averaging(patches, crops=mask_crops, shape=shape)
assert np.allclose(merged_mask, np.array([[4., 2., 0., 0.], [2., 1., 0., 0.], [0., 0., 0., 0.]]))
#%%
from src.eval.patch_based_eval.patch_logic import threshold_mask
final_mask = threshold_mask(merged_mask, nr_classes=1)
assert np.allclose(final_mask, np.array([[1., 1., 0., 0.], [1., 1., 0., 0.], [0., 0., 0., 0.]]))


#%%
from src.eval.visualization.visualize_imgs import plot_overlay_mask
plot_overlay_mask(merged_img, merged_mask)

#%%
import os
import SimpleITK as sitk
from src.eval.visualization.visualize_imgs import plot_overlay_mask

main_path = 'C:\\Users\\Matilda\\Nextcloud\\Data\\MedCom\\MedCom_resegmented'
patient = '017'

x = sitk.ReadImage(os.path.join(main_path, 'Pat_'+patient+'_img.mhd'))
y = sitk.ReadImage(os.path.join(main_path, 'Pat_'+patient+'_seg.mhd'))
x_slices = sitk.GetArrayFromImage(x)
y_slices = sitk.GetArrayFromImage(y)

img = x_slices[10]
mask = y_slices[10]

#%%
shape = img.shape
patches = generate_patches(img=img, patch_side=200, stride=100)
img_crops = [patch.crop(img=img) for patch in patches]
mask_crops = [patch.crop(img=mask) for patch in patches]
merged_img = merge_crops_overlapping(patches, crops=img_crops, shape=shape)


#%%
mask_crops[0] = 100
mask_crops[1] = 0
mask_crops[2] = 50
mask_crops[3] = 50
#%%
merged_mask = merge_crops_averaging(patches, crops=mask_crops, shape=shape)

plot_overlay_mask(merged_img, merged_mask)
