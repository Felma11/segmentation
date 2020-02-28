import os
from src.eval.visualization.confusion_matrix import ConfusionMatrix

def test_confusion_matrix():
    path = os.path.join('storage', 'tests')
    # We have 3 classes
    cm = ConfusionMatrix(3)
    # 2 tp for each class
    cm.add(predicted=0, actual=0, count=2)
    cm.add(predicted=1, actual=1, count=2)
    cm.add(predicted=2, actual=2, count=2)
    # 3 exampels of class 0 were predicted as class 1
    cm.add(predicted=1, actual=0, count=3)
    # 1 example of class 1 was predicted as class 2
    cm.add(predicted=2, actual=1, count=1)
    cm.plot(path, name='test_confusion_matrix' )
    assert cm.cm == [[2, 3, 0], [0, 2, 1], [0, 0, 2]]
    assert os.path.isfile(os.path.join(path, 'test_confusion_matrix.png'))