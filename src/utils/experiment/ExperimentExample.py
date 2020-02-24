from src.utils.experiment.Experiment import ExperimentRun

class ExperimentRunExample(ExperimentRun):
    # TODO: place within 'test directory'
    
    def __init__(self, root, dataset_ixs=None, name='', notes=''):
        super().__init__(root=root, dataset_ixs=dataset_ixs, name=name, notes=notes)
        self.summary_value = notes

    def write_summary_measures(self, results):
        print('WRITING')
        self.review['example key'] = self.summary_value