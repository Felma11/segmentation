import shutil
from src.utils.introspection import get_class

def test_introspection():
    class_path = 'src.utils.experiment.ExperimentExample.ExperimentExample'
    exp = get_class(class_path)(None, None)
    assert exp.__class__.__name__ == 'ExperimentExample'
    exp.finish()
    shutil.rmtree(exp.paths['root'])
