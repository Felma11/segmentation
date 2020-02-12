from src.eval.results import PartialResult

def test_partial_result():
    result = PartialResult(name='test', measure_names=['dice', 'ce'])
    result.add(epoch=0, measure_name='dice', value=0.2)
    result.add(epoch=0, measure_name='ce', value=2.0)
    result.add(epoch=5, measure_name='dice', value=0.5)
    result.add(epoch=10, measure_name='ce', value=0.5)
    result.add(epoch=15, measure_name='dice', value=0.7)
    result.add(epoch=15, measure_name='ce', value=0.1)
    assert result.get_epoch_measure(5, 'dice') == 0.5
    assert result.get_epoch_measure(5, 'ce') == None
    assert len(result.to_pandas()) == 4