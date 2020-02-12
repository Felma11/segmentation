from src.utils.experiment.Experiment import Experiment

class ExperimentExample(Experiment):
    
    def __init__(self, config, notes=None):
        super(ExperimentExample, self).__init__(config=config, notes=notes)

    def write_summary_measures(self, results):
        self.review['example key'] = 'example value'