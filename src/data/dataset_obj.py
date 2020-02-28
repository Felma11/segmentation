# ------------------------------------------------------------------------------
# This module contains the Instance and Dataset objects which store the path, 
# label and/or segmentation path for each image.
# Each experiment may have different splits for train and validation. However,
# the indexes which make up the test split are set at the beginning of working 
# with a dataset.
# ------------------------------------------------------------------------------

class Instance: 
    """
    An instance containing a path to x, a class value y and the path to a 
    segmentation mask.
    """
    def __init__(self, x_path, y=None, mask=None):
        assert (y is not None) or (mask is not None)
        self.x_path = x_path
        self.mask = mask
        self.y = y
        self.group_id = None

class Dataset:
    """A Dataset, which contains a list of instances."""
    def __init__(self, name, img_shape, file_type='png', nr_channels=3, 
        instances = [], hold_out_test_ixs = []):
        self.name = name
        self.file_type = file_type
        self.img_shape = img_shape
        self.nr_channels = nr_channels
        self.instances = instances
        self.size = len(self.instances)
        self.classes = set(ex.y for ex in instances)
        self.hold_out_test_ixs = hold_out_test_ixs
    
    def get_instances(self, index_list):
        return [x for (i, x) in enumerate(self.instances) if i in index_list]

    def pretty_print(self):
        class_dist = [len([e for e in self.examples[split] if e.y == c]) 
                for c in self.classes]
        string = ('Dataset ' + self.name + ' with classes: ' + str(self.classes) 
            + ', filetype: ' + self.file_type + '\n\r' 
            + str(len(self.instances))
            + 'Class distribution:'+str(class_dist)+'\n\r')
        print(string)

    def _get_class_instance_ixs(self, class_name, exclude_ixs=[]):
        return [ix for (ix, exp) in enumerate(self.instances) if exp.y == class_name and ix not in exclude_ixs]

    def _get_class_distribution(self, ixs):
        """
        :returns: a dictionary linking each class with the number of indexes
            that are examples with that class.
        """
        return {class_name: sum([1 if self.instances[ix].y==class_name else 0 for ix in ixs ]) for class_name in self.classes}