import numpy as np

class Accumulator:
    def __init__(self, keys=['loss']):
        self.values = dict()
        self.keys = keys
        self.init()

    def add(self, key, value, count=1):
        for i in range(count):
            self.values[key].append(value)

    def mean(self, key):
        return np.mean(self.values[key])

    def std(self, key):
        return np.std(self.values[key])

    def sum(self, key):
        return sum(self.values[key])

    def init(self):
        for key in self.keys:
            self.values[key] = []
