import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.autoencoding.autoencoder import Autoencoder

class PretrainedAutoencoder(Autoencoder):    
    def __init__(self, config, in_channels=3, img_size=(224, 224), hidden_dim=100):
        feature_model_name = config.get('feature_model_name', 'AlexNet')
        if feature_model_name == 'AlexNet':
            in_channels=3
            img_size=(224, 224)
            input_dim=9216

        super().__init__(config=config, in_channels=in_channels, img_size=img_size)

        self.feature_extractor = self.get_feature_extractor(feature_model_name)

        self.encoder_1 = nn.Linear(input_dim, hidden_dim)
        self.layers['encoder_1'] = self.encoder_1
        self.operations_after['encoder_1'] = [F.relu]

        self.first_decoder_layer_name = 'decoder_1'

        self.decoder_1 = nn.Linear(hidden_dim, input_dim)
        self.layers['decoder_1'] = self.decoder_1
        self.operations_after['decoder_1'] = [F.relu]

    def preprocess_input(self, x):
        '''
        Preprocessing that is done to the input before performing the 
        autoencoding, which is to say also to the target.
        '''
        # Instead of doing a forward pass, we exclude the classifier
        # See https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        x = self.feature_extractor.features(x)
        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        #sigmoid = nn.Sigmoid()
        #return sigmoid(x)
        return x

    def get_feature_extractor(self, feature_model_name='AlexNet'):
        '''
        AlexNet features are extracted from the input data. These are normalized 
        with the ImageNet statistics.
        '''
        # Fetch pretrained model
        if feature_model_name == 'AlexNet':  # input_size = 224
            feature_extractor = models.alexnet(pretrained=True)
        # Freeze pretrained parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False
        # Move to GPU
        feature_extractor.cuda()
        return feature_extractor
