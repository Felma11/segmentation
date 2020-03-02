# ------------------------------------------------------------------------------
# A class which accumulates results for easy plotting.
# 'PartialResult' stores the results for a sub-experiment, e.g. for a fold.
# 'ExperimentResult' calculates the average over all repetitions at the end.
# ------------------------------------------------------------------------------

import pandas as pd

def ExperimentResult():
    #TODO
    def __init__(self, metrics):
        pass

class PartialResult():
    def __init__(self, name, metrics):
        self.name = name
        self.metrics = metrics
        self.results = dict()

    def add(self, epoch, metric, value, split='train'):
        assert metric in self.metrics
        assert isinstance(epoch, int)
        assert isinstance(epoch, int)
        assert isinstance(value, float) or isinstance(value, int)
        if epoch not in self.results:
            self.results[epoch] = dict()
        if metric not in self.results[epoch]:
            self.results[epoch][metric] = dict()
        self.results[epoch][metric][split] = value

    def get_epoch_metric(self, epoch, metric, split='train'):
        try:
            value = self.results[epoch][metric][split]
            return value
        except:
            return None

    def to_pandas(self, metrics=None):
        if not metrics:
            metrics = self.metrics # Use all
        data = [[epoch, split, metric, 
            self.get_epoch_metric(epoch, metric, split=split)] 
            for epoch in self.results 
            for split in ['train', 'val', 'test'] 
            for metric in self.metrics if metric in metrics]
        data = [x for x in data if x[3]]
        df = pd.DataFrame(data, columns = ['Epoch', 'Split', 'Metric', 'Value'])
        return df
