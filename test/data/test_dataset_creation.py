import pytest

from src.data.datasets import mnist

@pytest.mark.skip(reason="Only test if MNIST dataset as .png in system")
def test_dataset_creation():
    root_path = 'C:\\Users\\cgonzale\\Documents\\data\\mnist_png' # TODO: adapt path
    ds = mnist(root_path=root_path, restore=False)
    assert len(ds.instances)==70000
    assert ds.hold_out_test_ixs[0]==60000
    assert ds.hold_out_test_ixs[-1]==69999
    assert len(ds.hold_out_test_ixs)==10000
    assert ds.classes == set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    




