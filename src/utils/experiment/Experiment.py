# ------------------------------------------------------------------------------
# Experiment class. The idea is that if multiple experiments are performed, and
# these should remain separate, all intermediate stored files and model states
# are stored within a directory for that experiment. In addition, the experiment
# directory contains the config.json file with the original configuration.
# Experiment names are timestamps for the time the experiment was created.
# ------------------------------------------------------------------------------

import os
import time
import sys
import numpy as np
import shutil

from src.utils.helper_functions import get_time_string
from src.utils.load_restore import join_path, pkl_dump, pkl_load, save_json

class Experiment:

    def __init__(self, config, notes=None):
        self.name = get_time_string()
        self.config = config
        # Create directories and assign to field
        self.paths = self._build_paths()
        # Save 'config.json' file
        save_json(self.config, self.paths['root'], 'config')
        # Set initial time
        self.time_start = time.time()
        # 'review.json' file
        self.review = dict()
        if notes:
            self.review['notes'] = notes

    def _build_paths(self):
        paths = dict()
        # Experiments are located in ./storage/experiments
        paths['root'] = join_path(['storage', 'experiments', self.name])
        # If the last experiment finished with an error, it is possible that
        # this has the same timestamp, so a '_' char is added at the end.
        while os.path.isdir(paths['root']):
            self.name += '_'
            paths['root'] = join_path(['storage', 'experiments', self.name])
        # Create root path
        os.makedirs(paths['root'])
        # Creates subdirectories for:
        # - agent_states: for model and optimizer state dictionaries
        # - results: for results files and visualizations
        # - obj: for all other files
        # - tmp: temporal files which are deleted after finishing the exp
        # Datasets should be experiment-independent, however, indexes for train
        # and validation sets should be stored in the experiment's 'obj'.
        for subpath in ['agent_states', 'obj', 'results', 'tmp']:
            paths[subpath] = os.path.join(paths['root'], subpath)
            os.mkdir(paths[subpath])
        return paths

    def update_review(self, dictionary):
        """Update 'review.json' file with external information."""
        for key, value in dictionary.items():
            self.review[key] = value

    def write_summary_measures(self, results):
        """Template method. write selected measures into the review."""

    def finish(self, results = None, exception = None):
        elapsed_time = time.time() - self.time_start
        self.review['elapsed_time'] = '{0:.2f}'.format(elapsed_time/60)
        if results:
            self.review['state'] = 'SUCCESS'
            pkl_dump(results, path=self.paths['results'], name='results')
            self.write_summary_measures(results)
        else:
            self.review['state'] = 'FAILED: ' + str(exception)
            # TODO: store exception with better format, or whole error path
        save_json(self.review, self.paths['root'], 'review')
        shutil.rmtree(self.paths['tmp'])

    