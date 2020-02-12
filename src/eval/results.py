# ------------------------------------------------------------------------------
# A class which accumulates results for easy plotting.
# 'PartialResult' stores the results for a sub-experiment, e.g. for a fold.
# 'ExperimentResult' calculates the average over all sub-experiments at the end.
# ------------------------------------------------------------------------------

import pandas as pd

def ExperimentResult():
    #TODO
    def __init__(self, measure_names):
        pass

class PartialResult():
    def __init__(self, name, measure_names):
        self.name = name
        self.measure_names = measure_names
        self.results = dict()

    def add(self, epoch, measure_name, value):
        assert measure_name in self.measure_names
        assert isinstance(epoch, int)
        assert isinstance(value, float) or isinstance(value, int)
        if epoch not in self.results:
            self.results[epoch] = dict()
        self.results[epoch][measure_name] = value

    def get_epoch_measure(self, epoch, measure_name):
        try:
            value = self.results[epoch][measure_name]
            return value
        except:
            return None

    def to_pandas(self):
        data = [[epoch]+[self.get_epoch_measure(epoch, measure_name) for measure_name in self.measure_names] for epoch in self.results]
        df = pd.DataFrame(data, columns = ['epoch'] + self.measure_names)
        return df
