
import shutil
from src.utils.experiment.Experiment import Experiment
from src.utils.load_restore import load_json, join_path

def test_experiment():
    config = {'cross_validation': True, 
        'nr_runs': 4, 
        'experiment_run_class_path': 'src.utils.experiment.ExperimentExample.ExperimentRunExample',
        'notes': 'Experiment notes.'}
    exp = Experiment(config=config)
    exp_path = exp.path
    exp_run = exp.get_experiment_run(idx_k=0, notes='Experiment run notes.')
    exp_run_path = exp_run.paths['root']
    exp_run.finish(results='not empty')
    config = load_json(path=exp_path, name='config')
    assert config['notes'] == 'Experiment notes.'
    review = load_json(path=exp_run_path, name='review')
    assert review['notes'] == 'Experiment run notes.'
    assert review['example key'] == 'Experiment run notes.'
    shutil.rmtree(exp_path)
