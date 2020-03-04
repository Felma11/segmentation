import numpy as np
import itertools

class Patch:
    def __init__(self, y_start, y_end, x_start, x_end):
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end

    def crop(self, img):
        return img[self.y_start:self.y_end,self.x_start:self.x_end]

    def zero_fill(self, crop, shape):
        """
        :returns: an image of the specified shape filled with zeros and the crop in
            the patch region
        """
        img = np.zeros(shape=shape)
        return self.paste(img=img, crop=crop)

    def paste(self, crop, img):
        img[self.y_start:self.y_end,self.x_start:self.x_end] = crop
        return img

    def patch_mask(self, shape):
        """
        :returns: a mask where the pixels of the patch are 1 and the rest 0
        """
        pixel_in_patch = np.zeros(shape=shape)
        pixel_in_patch[self.y_start:self.y_end,self.x_start:self.x_end] = 1
        return pixel_in_patch

    def __repr__(self):
        return str(self.__dict__)

def generate_patches(img, patch_side, stride):
    """Creates patches from left and right sides. When these cross, stop.
    Create one last patch in the middle region.
    """
    height, width = img.shape
    y_windows = _slide_window(side=height, window_side=patch_side, stride=stride)
    x_windows = _slide_window(side=width, window_side=patch_side, stride=stride)
    patches = list(itertools.product(y_windows, x_windows))
    patches = [Patch(y_start=p[0][0], y_end=p[0][1], x_start=p[1][0], x_end=p[1][1]) for p in patches]
    return patches

def _slide_window(side, window_side, stride):
    start = (0, window_side)
    end = (side-window_side, side)
    windows = [start, end]
    while start[1] <= end[0]:
        start = (start[0]+stride, start[1]+stride)
        end = (end[0]-stride, end[1]-stride)
        if start not in windows:
            windows.append(start)
        if end not in windows:
            windows.append(end)
    return windows

def merge_crops_overlapping(patches, crops, shape):
    assert len(patches) == len(crops)
    img = np.zeros(shape=shape)
    for i in range(len(patches)):
        img = patches[i].paste(crop=crops[i], img=img)
    return img

def merge_crops_averaging(patches, crops, shape):
    assert len(patches)==len(crops)
    zero_filled_imgs = []
    weights = []
    for i in range(len(patches)):
        zero_filled_imgs.append(patches[i].zero_fill(crop=crops[i], shape=shape))
        weights.append(patches[i].patch_mask(shape=shape))
    return np.average(zero_filled_imgs, axis=0, weights=weights)

def threshold_mask(mask, nr_classes=1):
    # TODO: extend for more classes
    return np.where(mask>0.5,1,0)

'''
import numpy as np
a = np.zeros((2,2))
a.fill(2)
c = np.zeros((3,3))
c[:2, :2] = a


pixel_in_crop = np.zeros((3,3))
crop_pixels[:2, :2] = 1

b = np.zeros((3,3))
b.fill(4)

d = np.average([b, c], axis=0, weights=[np.ones((3,3)), e])
'''