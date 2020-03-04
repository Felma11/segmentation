# ------------------------------------------------------------------------------
# The evaluation consists of cropping images in a sliding-window way and passing
# them through the network. 
# ------------------------------------------------------------------------------
def crop_image():
    pass


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



