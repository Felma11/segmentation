# ------------------------------------------------------------------------------
# Class all model definitions should descend from.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from cl.utils.load_restore import load_model_state, save_model_state

class Model(nn.Module):

    def __init__(self, model_config):
        super(Model, self).__init__()
        self.weights_file_name = model_config['weights_file_name']
        self.out_dim = model_config['out_dim']
        self.in_channels = model_config['in_channels']
        self.img_size = model_config['img_size']
        self.in_dim = self.in_channels*self.img_size[0]*self.img_size[1]

        # Pretrained features
        self.pretrained_features = lambda x: x

        # Initialization
        self.initialize(model_config['weights_file_name'])

    def forward(self, x):
        """ Template forward method."""
        pass

    def initialize(self, weights_file_name=None):
        """ Xavier initialization. For ReLu, He may be better."""
        if weights_file_name is not None:
            restored = load_model_state(self, weights_file_name, path = 'init_states')
            if restored:
                print('Initial parameters {} were restored'.format(weights_file_name))
            else:
                self.xavier_initialize()
                save_model_state(self, name=weights_file_name, path = 'init_states')
                print('Initial parameters {} were saved'.format(weights_file_name))
        else:
            self.xavier_initialize()

    def xavier_initialize(self):
        modules = [
            m for n, m in self.named_modules() if
            'conv' in n or 'linear' in n
        ]
        parameters = [
            p for
            m in modules for
            p in m.parameters() if
            p.dim() >= 2
        ]
        for p in parameters:
            nn.init.xavier_uniform_(p)
