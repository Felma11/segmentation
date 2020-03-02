from src.eval.results import PartialResult

def test_partial_result():
    result = PartialResult(name='test', metrics=['dice', 'ce'])
    result.add(epoch=0, metric='dice', value=0.2, split='train')
    result.add(epoch=0, metric='ce', value=2.0, split='train')
    result.add(epoch=0, metric='dice', value=0.3, split='val')
    result.add(epoch=0, metric='ce', value=2.5, split='val')
    result.add(epoch=5, metric='dice', value=0.5, split='train')
    result.add(epoch=10, metric='ce', value=0.5, split='train')
    result.add(epoch=5, metric='dice', value=0.6, split='val')
    result.add(epoch=10, metric='ce', value=0.7, split='val')
    result.add(epoch=15, metric='dice', value=0.7, split='train')
    result.add(epoch=15, metric='ce', value=0.1, split='train')
    result.add(epoch=15, metric='dice', value=0.8, split='val')
    result.add(epoch=15, metric='ce', value=0.15, split='val')
    assert result.get_epoch_metric(epoch=5, metric='dice', split='train') == 0.5
    assert result.get_epoch_metric(epoch=5, metric='ce', split='train') == None
    assert len(result.to_pandas()) == 12
    assert len(result.to_pandas(metrics=['dice'])) == 6