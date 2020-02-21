# ------------------------------------------------------------------------------
# This module contains the Instance and Dataset objects which store the path, 
# label and/or segmentation path for each image.
# Each experiment may have different splits for train and validation. However,
# the indexes which make up the test split are set at the beginning of working 
# with a dataset.
# ------------------------------------------------------------------------------

import random
import math

class Instance: 
    """
    An instance containing a path to x, a class value y and the path to a 
    segmentation mask.
    """
    def __init__(self, x_path, y=None, seg_path=None):
        assert (y is not None) or (seg_path is not None)
        self.x_path = x_path
        self.seg_path = seg_path
        self.y = y

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

    def _get_class_instance_ixs(self, class_name, exclude_ixs=[]):
        return [ix for (ix, exp) in enumerate(self.instances) if exp.y == class_name and ix not in exclude_ixs]

    def _get_class_distribution(self, ixs):
        """
        :returns: a dictionary linking each class with the number of indexes
            that are examples with that class.
        """
        return {class_name: sum([1 if self.instances[ix].y==class_name else 0 for ix in ixs ]) for class_name in self.classes}

    def split_instances(self, ratio=0.7, exclude_ixs=[], stratisfied=False):
        """
        Divides instances into two stratisfied sets. The stratification 
        operations prefers to give more examples of underrepresented classes
        to smaller sets (when the examples in a class cannot be split without
        a remainder).

        :param ratio: ratio of instances which remain in the first set.
        :param exclude: exclude these indexes from the splitting.
        :param stratisfied: should there be ca. as any examples for each class?
        :returns: two index lists with the indexes.
        """
        ixs = range(self.size)
        ixs = [ix for ix in ixs if ix not in exclude_ixs]
        nr_first_ds = math.floor(len(ixs)*ratio)
        if not stratisfied:
            random.shuffle(ixs)
            return ixs[:nr_first_ds], ixs[nr_first_ds:]
        else:
            ixs_1, ixs_2 = [], []
            class_instances = {class_name: self._get_class_instance_ixs(class_name=class_name, exclude_ixs=exclude_ixs) for class_name in self.classes}
            classes = list(self.classes)
            classes.sort(key=lambda x: len(class_instances[x]))
            for class_name in classes:
                exs = class_instances[class_name]
                random.shuffle(exs)
                # The mayority class is used to fill to the desired number of 
                # examples for each split
                if class_name == classes[-1]:
                    remaining_exs_nr = nr_first_ds - len(ixs_1)
                    if remaining_exs_nr == len(exs):
                        raise RuntimeError('Not enough examples of class {}'.format(class_name))
                    ixs_1 += exs[:remaining_exs_nr]
                    ixs_2 += exs[remaining_exs_nr:]
                # Otherwise, the operation makes sure less-represented classes
                # are as represented as possible in small sets
                else:
                    nr_class_first_ds = math.floor(len(exs)*ratio)
                    if nr_class_first_ds == len(exs):
                        raise RuntimeError('Not enough examples of class {}'.format(class_name))
                    ixs_1 += exs[:nr_class_first_ds]
                    ixs_2 += exs[nr_class_first_ds:]
        assert len(set(ixs_1+ixs_2+exclude_ixs)) == len(self.instances)
        return ixs_1, ixs_2 


    def _divide_sets_similar_length(self, exs, k):
        """
        Divides a list exs into k sets of similar length, with the initial ones 
        being longer.
        """
        random.shuffle(exs)
        nr_per_fold, remaining = divmod(len(exs), k)
        print('nr_per_fold '+str(nr_per_fold))
        print('remaining '+str(remaining))
        folds = []
        ix = 0
        for _ in range(k):
            nr_exs = nr_per_fold
            if remaining > 0:
                nr_exs += 1
            folds.append(exs[ix:ix+nr_exs])
            ix =+ ix+nr_exs
            remaining -= 1
        return folds



        '''
        nr_per_fold = len(exs) / k
        if nr_per_fold < 1:
            raise RuntimeError('Not enough examples.')
        nr_per_fold = mail.floor(nr_per_fold)
        remaining = len(exs) % k
        '''

    def create_instance_folds(self, k=5, exclude_ixs=[], stratisfied=False):
        """
        Divides instances into k stratisfied sets. Always, the most examples of 
        a class (when not divisible) are added to the fold that currently has
        the least examples.

        :param k: number of sets.
        :param exclude: exclude these indexes from the splitting
        :param stratisfied: should there be ca. as any examples for each class?
        :returns: k index lists with the indexes
        """
        ixs = range(self.size)
        ixs = [ix for ix in ixs if ix not in exclude_ixs]
        if not stratisfied:
            return self._divide_sets_similar_length(ixs, k)
        else:
            folds = [[] for k_ix in range(k)]
            class_instances = {class_name: self._get_class_instance_ixs(class_name=class_name, exclude_ixs=exclude_ixs) for class_name in self.classes}
            classes = list(self.classes)
            classes.sort(key=lambda x: len(class_instances[x]))
            for class_name in classes:
                exs = class_instances[class_name]
                # Sort so folds with least examples some first
                folds.sort(key=lambda x: len(x))
                divided_exs = self._divide_sets_similar_length(exs, k)
                print(divided_exs)
                for i in range(len(divided_exs)):
                    folds[i] += divided_exs[i]
        assert sum([len(fold) for fold in folds])+exclude_ixs == len(self.instances)
        return folds

def test_split_instances():
    instances = []
    instances.append(Instance('0A', y='0'))
    instances.append(Instance('1A', y='1'))
    instances.append(Instance('1B', y='1'))
    instances.append(Instance('0B', y='0'))
    instances.append(Instance('0C', y='0'))
    instances.append(Instance('1C', y='1'))
    instances.append(Instance('0D', y='0'))
    instances.append(Instance('0E', y='0'))
    ds = Dataset(name=None, img_shape=None, instances=instances)
    # Split at 70%
    ixs_1, ixs_2 = ds.split_instances(0.7, exclude_ixs=[3], stratisfied=True)
    class_dictribution = ds._get_class_distribution(ixs_1)
    assert class_dictribution['0'] == 2
    assert class_dictribution['1'] == 2
    class_dictribution = ds._get_class_distribution(ixs_2)
    assert class_dictribution['0'] == 2
    assert class_dictribution['1'] == 1
    # Split at 80%
    ixs_1, ixs_2 = ds.split_instances(0.8, exclude_ixs=[3], stratisfied=True)
    class_dictribution = ds._get_class_distribution(ixs_1)
    assert class_dictribution['0'] == 3
    assert class_dictribution['1'] == 2
    class_dictribution = ds._get_class_distribution(ixs_2)
    assert class_dictribution['0'] == 1
    assert class_dictribution['1'] == 1
    # Split at 90%
    # not possible because of too few examples of class 0
    try:
        ixs_1, ixs_2 = ds.split_instances(0.9, exclude_ixs=[3], stratisfied=True)
        assert False
    except RuntimeError:
        pass

def test_cross_validation():
    instances = []
    instances.append(Instance('0A', y='0'))
    instances.append(Instance('1A', y='1'))
    instances.append(Instance('1B', y='1'))
    instances.append(Instance('0B', y='0'))
    instances.append(Instance('0C', y='0'))
    instances.append(Instance('1C', y='1'))
    instances.append(Instance('0D', y='0'))
    instances.append(Instance('0E', y='0'))
    ds = Dataset(name=None, img_shape=None, instances=instances)
    # Split into 3 folds
    folds = ds.create_instance_folds(k=3, exclude_ixs=[3], stratisfied=True)
    print(folds)



test_cross_validation()