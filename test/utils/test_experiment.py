import shutil
from src.utils.experiment.ExperimentExample import ExperimentExample
from src.utils.load_restore import load_json, join_path

def test_introspection():
    config = {'model_config': {'model_name': 'cnn'}, 'agent_config': {'lr': 0.1}}
    exp = ExperimentExample(config, notes='An example experiment.')
    exp_name = exp.name
    exp.finish()
    exp_path = join_path(['storage', 'experiments', exp_name])
    review = load_json(path=exp_path, name='review')
    config = load_json(path=exp_path, name='config')
    assert review['notes'] == 'An example experiment.'
    assert config['model_config']['model_name'] == 'cnn'
    shutil.rmtree(exp.paths['root'])