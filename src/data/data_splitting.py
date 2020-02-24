# ------------------------------------------------------------------------------
# A dataset is usually split into train, validation and test sets. This
# splitting can be performed at random, for instance to perform n repetitions,
# or through cross-validation. Each repetition or fold is used for a different
# experiment run.
# In addition, a hold-out test dataset may be kept which is always the same and
# initialized together with a Dataset instance.
# ------------------------------------------------------------------------------

import random
import math

def split_dataset(dataset, test_ratio=0.2, val_ratio=0.2, nr_repetitions=5, cross_validation=True):
    """
    :param test_ratio: ratio of instances from 'dataset' for testing
    :param val_ratio: ratio of non-test instances from 'dataset' for validation
    :param nr_repetitions: number of times the experiment is repeated
    :param cross_validation: are the repetitions cross-val folds?
    :returns: A list of length 'nr_repetitions' where each item is a dictionary
        with keys 'train', 'val' and 'test'
    """
    splits = []
    if cross_validation:
        folds = create_instance_folds(dataset=dataset, k=nr_repetitions, 
            exclude_ixs=dataset.hold_out_test_ixs, stratisfied=True)
        for k in range(nr_repetitions):
            print('k {} of {}'.format(k, nr_repetitions))
            train, val = split_instances(dataset=dataset, ratio=1-val_ratio, 
                exclude_ixs=dataset.hold_out_test_ixs+folds[k],     
                stratisfied=True)
            splits.append({'train': train, 'val': val, 'test': folds[k]})
    else:
        for k in range(nr_repetitions):
            train_validation, test = split_instances(dataset=dataset, 
                ratio=1-test_ratio, exclude_ixs=dataset.hold_out_test_ixs, 
                stratisfied=True)
            train, val = split_instances(dataset=dataset, ratio=1-val_ratio, 
                exclude_ixs=dataset.hold_out_test_ixs+test, stratisfied=True)
            splits.append({'train': train, 'val': val, 'test': test})
    return splits

def split_instances(dataset, ratio=0.7, exclude_ixs=[], stratisfied=True):
    """
    Divides instances into two stratisfied sets. The stratification 
    operations prefers to give more examples of underrepresented classes
    to smaller sets (when the examples in a class cannot be split without
    a remainder).

    :param ratio: ratio of instances which remain in the first set.
    :param exclude: exclude these indexes from the splitting.
    :param stratisfied: should there be ca. as any examples for each class?
    :returns: two index lists with the.
    """
    ixs = range(dataset.size)
    ixs = [ix for ix in ixs if ix not in exclude_ixs]
    nr_first_ds = math.floor(len(ixs)*ratio)
    if not stratisfied:
        random.shuffle(ixs)
        return ixs[:nr_first_ds], ixs[nr_first_ds:]
    else:
        ixs_1, ixs_2 = [], []
        class_instances = {class_name: dataset._get_class_instance_ixs(
                class_name=class_name, exclude_ixs=exclude_ixs) for class_name 
                in dataset.classes}
        classes = list(dataset.classes)
        classes.sort(key=lambda x: len(class_instances[x]))
        for class_name in classes:
            exs = class_instances[class_name]
            random.shuffle(exs)
            # The mayority class is used to fill to the desired number of 
            # examples for each split
            if class_name == classes[-1]:
                remaining_exs_nr = nr_first_ds - len(ixs_1)
                if remaining_exs_nr == len(exs):
                    raise RuntimeError(
                        'Not enough examples of class {}'.format(class_name))
                ixs_1 += exs[:remaining_exs_nr]
                ixs_2 += exs[remaining_exs_nr:]
            # Otherwise, the operation makes sure less-represented classes
            # are as represented as possible in small sets
            else:
                nr_class_first_ds = math.floor(len(exs)*ratio)
                if nr_class_first_ds == len(exs):
                    raise RuntimeError(
                        'Not enough examples of class {}'.format(class_name))
                ixs_1 += exs[:nr_class_first_ds]
                ixs_2 += exs[nr_class_first_ds:]
    assert len(set(ixs_1+ixs_2+exclude_ixs)) == len(dataset.instances)
    return ixs_1, ixs_2 

def _divide_sets_similar_length(dataset, exs, k):
    """
    Divides a list exs into k sets of similar length, with the initial ones 
    being longer.
    """
    random.shuffle(exs)
    nr_per_fold, remaining = divmod(len(exs), k)
    if nr_per_fold < 1:
        raise RuntimeError('Not enough examples.')
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

def create_instance_folds(dataset, k=5, exclude_ixs=[], stratisfied=True):
    """
    Divides instances into k stratisfied sets. Always, the most examples of 
    a class (when not divisible) are added to the fold that currently has
    the least examples.

    :param k: number of sets.
    :param exclude: exclude these indexes from the splitting
    :param stratisfied: should there be ca. as any examples for each class?
    :returns: k index lists with the indexes
    """
    ixs = range(dataset.size)
    ixs = [ix for ix in ixs if ix not in exclude_ixs]
    if not stratisfied:
        return _divide_sets_similar_length(dataset, ixs, k)
    else:
        folds = [[] for k_ix in range(k)]
        class_instances = {class_name: dataset._get_class_instance_ixs(
                class_name=class_name, exclude_ixs=exclude_ixs) for 
                class_name in dataset.classes}
        classes = list(dataset.classes)
        classes.sort(key=lambda x: len(class_instances[x]))
        for class_name in classes:
            exs = class_instances[class_name]
            # Sort so folds with least examples some first
            folds.sort(key=lambda x: len(x))
            divided_exs = _divide_sets_similar_length(dataset, exs, k)
            for i in range(len(divided_exs)):
                folds[i] += divided_exs[i]
    assert sum([len(fold) for fold in folds])+len(exclude_ixs) == len(dataset.instances)
    return folds