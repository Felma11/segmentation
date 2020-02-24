import shutil
from src.utils.introspection import get_class
from src.utils.load_restore import join_path

def test_introspection():
    class_path = 'src.utils.experiment.ExperimentExample.ExperimentRunExample'
    root = join_path(['storage', 'experiments', 'test_introspection'])
    exp = get_class(class_path)(root, notes='test value')
    assert exp.__class__.__name__ == 'ExperimentRunExample'
    shutil.rmtree(join_path([root, exp.name]))
