# ------------------------------------------------------------------------------
# This module contains the Instance and Dataset objects which store the path, 
# label and/or segmentation path for each image.
# ------------------------------------------------------------------------------
 
class Dataset:
    """A Dataset, which contains a list of instances."""
    def __init__(self, name, file_type, img_shape, nr_channels, 
        classes = None, instances = []):
        self.name = name
        self.file_type = file_type
        self.img_shape = img_shape
        self.nr_channels = nr_channels
        self.instances = instances
        self.classes = classes
    
    def get_examples(self, index_list):
        instances = []
        return self.examples[split]

    def pretty_print(self):
        class_dist = [len([e for e in self.examples[split] if e.y == c]) 
                for c in self.classes]
        string = ('Dataset ' + self.name + ' with classes: ' + str(self.classes) 
            + ', filetype: ' + self.file_type + '\n\r' 
            + str(len(self.instances))
            + 'Class distribution:'+str(class_dist)+'\n\r')
        print(string)

class Instance: 
    """An instance containing a path to x, a class value y and the path to a segmentation mask."""
    def __init__(self, x_path, y=None, seg_path=None):
        assert (y is not None) or (seg_path is not None)
        self.x_path = x_path
        self.seg_path = seg_path
        self.y = y
