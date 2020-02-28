import numpy as np
import torch
import torch.nn as nn

from cl.autoencoders.train_autoencoder import train_or_load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, Lambda, GlobalAveragePooling2D
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()

from cl.utils.load_restore import join_path

from cl.oracles.prostate_oracle import ProstateOracle
from cl.utils.load_restore import estimation_save, estimation_load

def log_prob(x, mu, sigma_inv, eigvals):
    # Evaluate a multivariate gaussian PDF with mean mu and covariance matrix sigma
    # i.e. P(x | domain)
    # = N(x, mu, sigma)
    # Calculate density in log space because the eigenvalues are too small and the determinant would be 0 otherwise.
    # @ is for matrix multiplication
    
    # this is also a degenerate case where some of the eigenvalues of sigma are 0
    # so we take the pseudo-inverse and the product of the eigenvals greater than 0
    log_inv_det=-np.sum(np.log(2*np.pi*eigvals[eigvals>0]))
    log_exponent=-(1/2)*(x-mu).T@sigma_inv@(x-mu)
    return log_inv_det+log_exponent

from cl.autoencoders.cnn_autoencoder_small import CNNAutoencoderSmall
from cl.autoencoders.pretrained_autoencoder import PretrainedAutoencoder
autoencoders = {'CNNAutoencoder': CNNAutoencoderSmall, 'PretrainedAutoencoder': PretrainedAutoencoder}

class ProstateDensityOracle(ProstateOracle):
    def __init__(self, dataset_name, experiment_path, feature_model_name='InceptionV3', saved_model_name=None, batch_size=1):
        super(ProstateDensityOracle, self).__init__(dataset_name, experiment_path, batch_size=batch_size, lowest_score=False, name='ProstateDensityOracle')

        self.feature_model = self.get_feature_model(feature_model_name, saved_model_name)
        saved_params = estimation_load(dataset_name, feature_model_name=feature_model_name)
        if saved_params is None:
            train_datasets = [torch.utils.data.ConcatDataset([self.datasets['train'][task_ix], self.datasets['val'][task_ix]]) for task_ix in range(self.nr_tasks)]
            print('Getting initial features')
            train_features = [self.get_features(torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)) for train_dataset in train_datasets]
            print('Calculating initial values for estimation')
            self.mus, self.sigmas, self.sigma_eigenvals, self.sigma_invs, self.biases = self.get_density_information(train_features)
            estimation_save((self.mus, self.sigmas, self.sigma_eigenvals, self.sigma_invs, self.biases), dataset_name, feature_model_name=feature_model_name)
        else:
            self.mus, self.sigmas, self.sigma_eigenvals, self.sigma_invs, self.biases = saved_params

    def set_scores(self, dl_task, split='val'):
        '''
        Calculate a score for each model for each example in the dataset
        '''
        if split=='train':
            dataloader = self.train_loaders[dl_task]
        elif split=='val':
            dataloader = self.val_loaders[dl_task]
        else:
            assert split=='test'
            dataloader = self.test_loaders[dl_task]
        features = self.get_features(dataloader)
        for i in range(len(features)):
            x = features[i]
            domains_score = self.predict_domain(x)
            for model_task_ix in range(self.nr_tasks):
                self.scores[split][dl_task][model_task_ix].append(domains_score[model_task_ix])

    def get_feature_model(self, feature_model_name, saved_model_name = 'PretrainedAutoencoder_3'):
        if 'Autoencoder' in feature_model_name:
            autoencoder = autoencoders['CNNAutoencoder'](input_channels=3)
            autoencoder.cuda()
            criterion = nn.MSELoss()
            train_or_load(autoencoder, self.dataset_name, 1, criterion, self.train_loaders[1], self.val_loaders[0])
            autoencoder.load_encoder_decoder(self.dataset_name, 1)
            return autoencoder.encoder
        if feature_model_name == 'DenseNet':
            big_model=keras.applications.DenseNet121(input_shape=(32,32,3), include_top=False, weights="imagenet")
            feature_model_ip=Input(shape=(32,32,3))
            feature_model=feature_model_ip
            feature_model=big_model(feature_model)
            feature_model=GlobalAveragePooling2D()(feature_model)
            feature_model=Model(feature_model_ip,feature_model)
            return feature_model
        if feature_model_name=='InceptionV3':
            big_model=keras.applications.InceptionV3(input_shape=(299,299,3), include_top=False, weights="imagenet")
            feature_model_ip=Input(shape=(299,299,3))
            feature_model=feature_model_ip
            feature_model=big_model(feature_model)
            feature_model=GlobalAveragePooling2D()(feature_model)
            feature_model=Model(feature_model_ip, feature_model)
            return feature_model

    def get_features(self, dataloader):
        print('getting features')
        print(len(dataloader))
        i = 0
        task_x_features = []
        for x, y in dataloader:
            i += 1
            #print('x shape: {}'.format(x.shape)) x shape: torch.Size([1, 3, 299, 299])
            imgs = x.numpy()
            #print('imgs shape: {}'.format(imgs.shape)) imgs shape: (1, 3, 299, 299)
            imgs = np.rollaxis(imgs, 1, 4)
            #print('imgs shape: {}'.format(imgs.shape)) imgs shape: (1, 299, 299, 3)
            try:
                # TensorFlow model
                features = self.feature_model.predict(imgs, verbose=0)
                #print('Features shape: {}'.format(features.shape)) Features shape: (1, 2048)
                task_x_features.append(features)
            except:
                # PyTorch model
                #print(imgs.shape) == (1, 299, 299, 3)
                features = self.feature_model(x.cuda()).detach().cpu().numpy().reshape((1, -1))
                print('Features shape: {}'.format(features.shape))
                #print(features.shape) torch.Size([1, 16, 291, 291])
                task_x_features.append(features)
        print('All features shape: {}'.format(np.concatenate(task_x_features).shape))
        #print(np.concatenate(task_x_features).shape) == (len(dataloader), feature_size)
        return np.concatenate(task_x_features)

    def get_density_information(self, x_features):
        mus=[]
        for i in range(self.nr_tasks):
            mus.append(np.mean(x_features[i],axis=0))
        #assert mus[0].shape == (1024,)

        sigmas=[]
        for i in range(self.nr_tasks):
            sigmas.append(np.cov(x_features[i],rowvar=0))
        #assert sigmas[0].shape == (1024, 1024)

        sigma_eigenvals=[]
        for i in range(self.nr_tasks):
            sigma_eigenvals.append(np.linalg.eigvals(sigmas[i]).real)

        sigma_invs=[]
        for i in range(self.nr_tasks):
            sigma_invs.append(np.linalg.pinv(sigmas[i]))

        biases=[]
        for k in range(self.nr_tasks):
            biases.append(log_prob(mus[k], mus[k], sigma_invs[k], sigma_eigenvals[k]))

        return mus, sigmas, sigma_eigenvals, sigma_invs, biases

    def predict_domain(self, x):
        # predict domain by taking P(x | domain), 
        # i.e. N(x | mu_domain, sigma_domain) / N(mu_domain | mu_domain, sigma_domain)
        # = log(N(x | mu_domain, sigma_domain)) - log(N(mu_domain | mu_domain, sigma_domain))
        
        # the bias term is strictly speaking not correct from a probability standpoint, 
        # but helps empirically, because the densities are degenerate.
        
        densities=[]
        for k in range(self.nr_tasks):
            densities.append(log_prob(x, self.mus[k], self.sigma_invs[k], self.sigma_eigenvals[k])- self.biases[k])
            #densities.append(-np.sqrt(np.mean(np.abs(x-mus[k]))))
        densities=np.array(densities)
        return densities