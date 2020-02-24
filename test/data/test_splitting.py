
from src.data.dataset_obj import Dataset, Instance
from src.data.data_splitting import split_instances, create_instance_folds, split_dataset

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
    ixs_1, ixs_2 = split_instances(ds, 0.7, exclude_ixs=[3], stratisfied=True)
    class_dictribution = ds._get_class_distribution(ixs_1)
    assert class_dictribution['0'] == 2
    assert class_dictribution['1'] == 2
    class_dictribution = ds._get_class_distribution(ixs_2)
    assert class_dictribution['0'] == 2
    assert class_dictribution['1'] == 1
    # Split at 80%
    ixs_1, ixs_2 = split_instances(ds, 0.8, exclude_ixs=[3], stratisfied=True)
    class_dictribution = ds._get_class_distribution(ixs_1)
    assert class_dictribution['0'] == 3
    assert class_dictribution['1'] == 2
    class_dictribution = ds._get_class_distribution(ixs_2)
    assert class_dictribution['0'] == 1
    assert class_dictribution['1'] == 1
    # Split at 90%
    # not possible because of too few examples of class 0
    try:
        ixs_1, ixs_2 = split_instances(ds, 0.9, exclude_ixs=[3], stratisfied=True)
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
    # Split into 2 folds
    folds = create_instance_folds(ds, k=2, exclude_ixs=[3], stratisfied=True)
    for fold in folds:
        assert len(fold) < 5
        class_dictribution = ds._get_class_distribution(fold)
        assert class_dictribution['1']==1 or class_dictribution['1']==2
        assert class_dictribution['0']==2
    # Split into 3 folds
    folds = create_instance_folds(ds, k=3, exclude_ixs=[3], stratisfied=True)
    for fold in folds:
        assert len(fold) < 4
        class_dictribution = ds._get_class_distribution(fold)
        assert class_dictribution['1']==1 
        assert class_dictribution['0']==1 or class_dictribution['0']==2
    # Split into 4 folds
    try:
        folds = create_instance_folds(ds, k=4, exclude_ixs=[3], stratisfied=True)
        assert False
    except RuntimeError:
        pass

def test_split_dataset():
    instances = []
    instances.append(Instance('0A', y='0'))
    instances.append(Instance('0B', y='0'))
    instances.append(Instance('0C', y='0'))
    instances.append(Instance('0D', y='0'))
    instances.append(Instance('0E', y='0'))
    instances.append(Instance('1A', y='1'))
    instances.append(Instance('1B', y='1'))
    instances.append(Instance('1C', y='1'))
    instances.append(Instance('1D', y='1'))
    instances.append(Instance('1E', y='1'))
    instances.append(Instance('1F', y='1'))
    instances.append(Instance('1G', y='1'))
    instances.append(Instance('1H', y='1'))
    instances.append(Instance('0A2', y='0'))
    instances.append(Instance('0B2', y='0'))
    instances.append(Instance('0C2', y='0'))
    instances.append(Instance('0D2', y='0'))
    instances.append(Instance('0E2', y='0'))
    instances.append(Instance('1A2', y='1'))
    instances.append(Instance('1B2', y='1'))
    instances.append(Instance('1C2', y='1'))
    instances.append(Instance('1D2', y='1'))
    instances.append(Instance('1E2', y='1'))
    instances.append(Instance('1F2', y='1'))
    instances.append(Instance('1G2', y='1'))
    instances.append(Instance('1H2', y='1'))
    ds = Dataset(name=None, img_shape=None, instances=instances)
    # Split into 2 folds. Then split into train, test and validation sets
    folds = split_dataset(ds, test_ratio=0.2, val_ratio=0.2, nr_repetitions=2, cross_validation=True)
    for fold in folds:
        assert len(set(fold['train'] + fold['val'] + fold['test'])) == len(fold['train'] + fold['val'] + fold['test'])
        class_dictribution = ds._get_class_distribution(fold['train'])
        assert class_dictribution['0']==4
        assert class_dictribution['1']==6
        class_dictribution = ds._get_class_distribution(fold['val'])
        assert class_dictribution['0']==1
        assert class_dictribution['1']==2
        class_dictribution = ds._get_class_distribution(fold['test'])
        assert class_dictribution['0']==5
        assert class_dictribution['1']==8
    # Repetitions
    splits = split_dataset(ds, test_ratio=0.2, val_ratio=0.2, nr_repetitions=3, cross_validation=False)
    for split in splits:
        class_dictribution = ds._get_class_distribution(split['train'])
        assert class_dictribution['0']==6
        assert class_dictribution['1']==10
        class_dictribution = ds._get_class_distribution(split['val'])
        assert class_dictribution['0']==2
        assert class_dictribution['1']==2
        class_dictribution = ds._get_class_distribution(split['test'])
        assert class_dictribution['0']==2
        assert class_dictribution['1']==4