
#%%

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def plot_3d_img(img):
    """
    :param img: SimpleITK image or numpy array
    """
    if 'SimpleITK.SimpleITK.Image' in str(type(img)):
        img = sitk.GetArrayFromImage(img)
    assert len(img.shape) == 3
    assert img.shape[1] == img.shape[2]
    plt.figure(figsize=(20,16))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(img.shape[0]):
        plt.subplot(5,6,i+1), plt.imshow(img[i]), plt.axis('off')
    plt.show()

def plot_3d_segmentation(img, segmentation):
    """
    :param img: SimpleITK image or numpy array
    """
    if 'SimpleITK.SimpleITK.Image' in str(type(img)):
        img = sitk.GetArrayFromImage(img)
        segmentation = sitk.GetArrayFromImage(segmentation)
    assert len(img.shape) == 3
    assert img.shape[1] == img.shape[2] # Channels first
    assert img.shape == segmentation.shape
    
    plt.figure(figsize=(20,16))
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(img.shape[0]):
        plt.subplot(5,6,i+1), plt.imshow(img[i], 'gray', interpolation='none'), plt.axis('off')
        plt.subplot(5,6,i+1), plt.imshow(segmentation[i], 'jet', interpolation='none', alpha=0.5), plt.axis('off')
    plt.show()

def plot_overlay_mask(img, mask):
    if 'torch' in str(type(img)):
        img, mask = img.cpu().numpy(), mask.cpu().numpy()
        while len(img.shape) > 2:
            img, mask = img[0], mask[0]
    assert img.shape == mask.shape
    plt.figure()
    plt.imshow(img, 'gray')
    plt.imshow(mask, 'jet', alpha=0.7)
    plt.show()

def compare_masks(gt_mask, pred_mask):
    assert gt_mask.shape == pred_mask.shape
    plt.figure()
    plt.imshow(gt_mask, 'gray')
    plt.imshow(pred_mask, 'jet', alpha=0.7)
    plt.show()

#%%
'''
path = 'C:\\Users\\cgonzale\\Documents\\data\\MedCom_resegmented'
preds = 'C:\\Users\\cgonzale\\Documents\\data\\preds'
patient = '016'

#'016', '022', '027', '031', '041', '114'

x = sitk.ReadImage(os.path.join(root_path, 'Pat_'+patient+'_img.mhd'))
y = sitk.ReadImage(os.path.join(root_path, 'Pat_'+patient+'_seg.mhd'))
y_pred = sitk.ReadImage(os.path.join(root_path, 'Pat_'+patient+'_pred.nii.gz'))

# Visualize from dataloader
from src.eval.visualization.visualize_imgs import plot_overlay_mask
dataloader = dataloaders['train']
for x, y in dataloader:
    for i in range(len(x)):
        img, mask = x[i], y[i]
        if torch.nonzero(mask).size(0) > 0:
            plot_overlay_mask(img, mask)
    break
# %%
'''