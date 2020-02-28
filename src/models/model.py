# ------------------------------------------------------------------------------
# Class all model definitions should descend from.
# ------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary  import summary
from collections import OrderedDict

from src.utils.pytorch.pytorch_load_restore import load_model_state, save_model_state

class Model(nn.Module):
    """
    :param in_channels: input channels
    :param img_size: (x_size, y_size) tuple
    """
    def __init__(self, config, in_channels=3, img_size=(299,299)):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.weights_file_name = config.get('weights_file_name', None)
        self.out_dim = config.get('out_dim', None) # For classification

        self.input_shape = (self.in_channels, self.img_size[0], self.img_size[1])

        # E.g. pretrained features
        def preprocess_input(self, x):
            return x

        # Initialization
        self.initialize(self.weights_file_name)

        # Every subclass of the 'model' class has the following lists defined
        self.layers = OrderedDict() # Definition of all layers
        self.operations_before = dict() # Performed before a layer
        # E.g. x.view(-1, self.num_flat_features(x)
        self.operations_after = dict() # Performed after a layer

    def forward(self, x):
        for layer_name, layer in self.layers.items():
            for operation in self.operations_before.get(layer_name, []):
                x = operation(x)
            x = layer(x)
            for operation in self.operations_after.get(layer_name, []):
                x = operation(x)
        return x

    # Initialization

    def initialize(self, weights_file_name=None):
        """
        Tries to restore a previous model. If no model is found but a file name
        if provided, the model is saved.
        """
        if weights_file_name is not None:
            path = os.path.join('storage', 'agent_states')
            restored = load_model_state(self, weights_file_name, path=path)
            if restored:
                print('Initial parameters {} were restored'.format(weights_file_name))
            else:
                self.xavier_initialize()
                save_model_state(self, name=weights_file_name, path=path)
                print('Initial parameters {} were saved'.format(weights_file_name))
        else:
            self.xavier_initialize()

    def xavier_initialize(self):
        """Xavier initialization. For ReLu, He may be better."""
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

    def get_param_list_static(self):
        """
        Returns a 1D array of parameter values
        """
        model_params_array = []
        for _, param in self.state_dict().items():
            model_params_array.append(param.reshape(-1).cpu().numpy())
        return np.concatenate(model_params_array)


    # Methods to calculate the feature size

    def num_flat_features(self, x):
        """
        Flattened view of all dimensions except the batch size.
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def size_before_lin(self, shape_input):
        """ 
        Size after linearization
        returns: integer of dense size
        """
        return shape_input[0]*shape_input[1]*shape_input[2]

    def size_after_conv(self, shape_input, output_channels, kernel):
        """
        Gives the number of output neurons after the conv operation.
        The first dimension is the channel depth and the other 2 are given by
        input volume (size - kernel size + 2*padding)/stride + 1
        param shape_input: (nr_channels, dim_x, dim_y)
        returns: 3D tuple
        """
        return (output_channels, shape_input[1]-kernel+1, shape_input[2]-kernel+1)

    def size_after_pooling(self, shape_input, shape_pooling):
        """
        Maintains the first input dimension, which is the output channels in 
        the previous conv layer. The others are divided by the shape of the 
        pooling.
        """
        return (shape_input[0], shape_input[1]//shape_pooling[0], shape_input[2]//shape_pooling[1])


    # Methods to output model information

    def model_summary(self):
        """Keras-style summary."""
        summary(self, input_size=(3, 299, 299))

    def model_description(self):
        print('Model: {}\n'.format(self.__class__))

        features_size = self.input_shape
        print('Input shape (channels first): {}\n'.format(features_size))

        state_dictionary = self.state_dict()
        for layer_name, layer in self.layers.items():
            layer_type = layer.__class__.__name__
            print('{} layer {}'.format(layer_type, layer_name))
            weights = state_dictionary.get(layer_name+'.weight', None)
            bias = state_dictionary.get(layer_name+'.bias', None)
            if weights is not None and bias is not None:
                print('Shape weights: {}, bias: {}'.format(list(weights.shape), list(bias.shape)))