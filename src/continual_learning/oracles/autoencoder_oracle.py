import os
import torch
import torch.nn as nn

from src.continual_learning.oracles.oracle import Oracle
from src.utils.introspection import get_class

class AutoencoderOracle(Oracle):
    def __init__(self, pytorch_datasets, save_path, feature_model_name='AlexNet', autoencoder_path='', nr_training_epochs = 20):
        autoencoder_name = autoencoder_path.split('.')[-1]
        super().__init__(pytorch_datasets=pytorch_datasets, save_path=save_path, batch_size=1, lowest_score=True, name='AutoencoderOracle_'+autoencoder_name+'_'+feature_model_name)
        
        self.autoencoder_path = autoencoder_path
        self.feature_model_name = feature_model_name
        self.nr_training_epochs = nr_training_epochs

        # Change the data transform
        for task_ix in range(self.nr_tasks):
            for split in self.splits:
                self.datasets[task_ix][split].set_tranform(transform=feature_model_name)

    def train_or_load(self, task_ix):
        autoencoder = get_class(self.autoencoder_path)(config={'feature_model_name': self.feature_model_name})
        autoencoder.cuda()
        optimizer = torch.optim.Adam(autoencoder.parameters(), weight_decay=1e-4)
        criterion = nn.MSELoss()
        agent = get_class('src.agents.autoencoder_agent.AutoencoderAgent')(config={'tracking_interval': self.nr_training_epochs//5}, model=autoencoder, optimizer=optimizer, criterion=criterion)
        path = os.path.join(self.save_path, 'agent_states', self.name)
        name = 'task_'+str(task_ix)
        if agent.restore_state(path=path, name=name):
            print('Previous model state restored')    
        else:
            print('Training autoencoder for task {}'.format(task_ix))
            agent.train_model(dataloaders=self.dataloaders[task_ix], nr_epochs=20)
            agent.save_state(path=path, name=name)
        return agent

    def set_scores(self, dl_task, split='val'):
        '''
        Calculate a score for each model for each example in the dataset
        '''
        dataloader = self.dataloaders[dl_task][split]
        for model_task_ix in range(self.nr_tasks):
            agent = self.train_or_load(task_ix=model_task_ix)
            for data_batch in dataloader:
                inputs, targets, outputs = agent.get_input_target_output(data_batch)
                loss = agent.calculate_criterion(outputs, targets)
                print(float(loss))
                print(loss.cpu().numpy())
                self.scores[split][dl_task][model_task_ix].append(loss.cpu().numpy())