import pytest

from src.data.datasets import mnist
from src.data.data_splitting import split_dataset

@pytest.mark.skip(reason="Only test if MNIST dataset as .png in system")
def test_dataset_creation():
    root_path = 'C:\\Users\\cgonzale\\Documents\\data\\mnist_png' # TODO: adapt path
    ds = mnist(root_path=root_path, restore=False)
    folds = split_dataset(ds, val_ratio=0.2, nr_repetitions=10, cross_validation=True)
    for fold in folds:
        assert len(set(fold['train'] + fold['val'] + fold['test'])) == len(ds.instances) - len(ds.hold_out_test_ixs)
        class_dictribution = ds._get_class_distribution(fold['train'])
        for class_name in ds.classes:
            4500 > class_dictribution[class_name] > 3800
        class_dictribution = ds._get_class_distribution(fold['val'])
        for class_name in ds.classes:
            1300 > class_dictribution[class_name] > 900
        class_dictribution = ds._get_class_distribution(fold['test'])
        for class_name in ds.classes:
            700 > class_dictribution[class_name] > 500