from src.utils.accumulator import Accumulator

def test_accumulator():
    acc = Accumulator(keys=['loss', 'acc'])
    acc.add('loss', 0.2)
    acc.add('acc', 0.7)
    acc.add('loss', 0.1)
    acc.add('acc', 0.9)
    assert 0.1499 < acc.mean('loss') < 0.1501
    assert 0.799 < acc.mean('acc') < 0.801
    assert 0.049 < acc.std('loss') < 0.0501
    assert 0.099 < acc.std('acc') < 0.1001
    assert 0.299 < acc.sum('loss') < 0.301
    assert 1.599 < acc.sum('acc') < 1.601
    acc.init()
    acc.add('loss', 0.1)
    assert 0.0999 < acc.mean('loss') < 0.101