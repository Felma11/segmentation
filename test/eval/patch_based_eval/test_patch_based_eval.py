import numpy as np
from src.eval.patch_based_eval.patch_logic import _slide_window, generate_patches, merge_crops_overlapping, merge_crops_averaging, threshold_mask

def test_patch_based_utils():
    assert set(_slide_window(side=400, window_side=200, stride=100)) == {(0, 200), (100, 300), (200, 400)}
    assert set(_slide_window(side=400, window_side=300, stride=200)) == {(0, 300), (100, 400)}
    assert set(_slide_window(side=400, window_side=100, stride=100)) == {(0, 100), (100, 200), (200, 300), (300, 400)}
    assert set(_slide_window(side=400, window_side=200, stride=50)) == {(0, 200), (50, 250), (150, 350), (200, 400)}
    assert set(_slide_window(side=300, window_side=200, stride=100)) == {(0, 200), (100, 300)}
    assert set(_slide_window(side=300, window_side=300, stride=200)) == {(0, 300)}
    assert set(_slide_window(side=300, window_side=100, stride=100)) == {(0, 100), (100, 200), (200, 300)}
    assert set(_slide_window(side=300, window_side=200, stride=50)) == {(0, 200), (100, 300)}

def test_patch_based_eval():
    img = np.zeros((3, 4))
    mask = np.zeros((3, 4))
    shape = img.shape
    patches = generate_patches(img=mask, patch_side=2, stride=1)
    img_crops = [patch.crop(img=img) for patch in patches]
    mask_crops = [patch.crop(img=mask) for patch in patches]
    assert len(patches) == len(img_crops) == len(mask_crops) == 6
    img_crops[0].fill(2)
    merged_img = merge_crops_overlapping(patches, crops=img_crops, shape=shape)
    assert np.allclose(merged_img, np.array([[2., 2., 0., 0.], [2., 2., 0., 0.], [0., 0., 0., 0.]]))
    mask_crops[0]=4
    merged_mask = merge_crops_averaging(patches, crops=mask_crops, shape=shape)
    assert np.allclose(merged_mask, np.array([[4., 2., 0., 0.], [2., 1., 0., 0.], [0., 0., 0., 0.]]))
    final_mask = threshold_mask(merged_mask, nr_classes=1)
    assert np.allclose(final_mask, np.array([[1., 1., 0., 0.], [1., 1., 0., 0.], [0., 0., 0., 0.]]))
