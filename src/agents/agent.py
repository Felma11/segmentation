# ------------------------------------------------------------------------------
# Agents contain models, which they train, as well as additional functionality.
# ------------------------------------------------------------------------------

import os
from src.utils.accumulator import Accumulator
from src.utils.pytorch.pytorch_load_restore import load_model_state, save_model_state, save_optimizer_state, load_optimizer_state, save_scheduler_state, load_scheduler_state
import sys

class Agent():
    def __init__(self, config, exp_paths, model, scheduler, optimizer, 
        results=None, criterion=None, agent_name=''):
        self.agent_name = agent_name
        self.exp_paths = exp_paths
        self.config = config

        self.model = model
        self.acc = Accumulator()
        self.results = results

        # TODO: initialize scheduler, optimizer and criterion based on the config
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion_fn = criterion

        # Continue traning model
        '''

        for name, param in self.model.state_dict().items():
            print(name)
            print(param.shape)
        sys.exit()
        '''
        self.start_epoch = 0
        restart_from_latest = self.config.get('restart_from_latest', True)    
        if restart_from_latest:
            model_states = [f for f in os.listdir(self.exp_paths['agent_states']) if f.split('_')[0]==self.agent_name]
            if model_states:
                epochs = [int(model_state.split('_')[1]) for model_state in model_states]
                latest_epoch = max(epochs)
                self.start_epoch = latest_epoch
                # TODO: use name argument
                self.restore_state(epoch=latest_epoch)

    def save_state(self, epoch, name=''):
        """Save all necessary to recuperate the state."""
        path = self.exp_paths['agent_states']
        state_name = self.agent_name + '_' + str(epoch)
        if name:
            state_name += '_' + name
        save_model_state(self.model, name=state_name, path=path)
        save_optimizer_state(self.optimizer, name=state_name, path=path)
        save_scheduler_state(self.scheduler, name=state_name, path=path)

    def restore_state(self, epoch, name=''):
        path = self.exp_paths['agent_states']
        state_name = self.agent_name + '_' + str(epoch)
        if name:
            state_name += '_' + name
        restored_model = load_model_state(self.model, name=state_name, path=path)
        restored_optimizer = load_optimizer_state(self.optimizer, name=state_name, path=path)
        restored_scheduler = load_scheduler_state(self.scheduler, name=state_name, path=path)
        if restored_model and restored_optimizer and restored_scheduler:
            print('Restored state {}'.format(state_name))
        return restored_model and restored_optimizer and restored_scheduler

    def eval(self, dataloader, measures=['loss']):
        self.model.train(False)
        val_acc = Accumulator(keys=measures)
        for i, data in enumerate(dataloader):
            inputs, targets, outputs = self.get_input_target_output(data)
            if 'loss' in measures:
                loss = self.calculate_criterion(outputs, targets)
                val_acc.add('loss', loss.item())
        return val_acc

    def track_statistics(self, epoch, dataloaders):
        # TODO: store running_loss
        # TODO track and print other measures every so many epochs
        if epoch % self.config.get('tracking_interval', 1) == 0:
            print('Epoch {}'.format(epoch))
            print('Avg. train loss accumulated: {}'.format(self.acc.mean('loss')))
            train_values = self.eval(dataloader=dataloaders['train'], measures=['loss'])
            print('Avg. train loss: {}'.format(train_values.mean('loss')))
            validation_values = self.eval(dataloader=dataloaders['val'], measures=['loss'])
            print('Avg. validation loss: {}'.format(validation_values.mean('loss')))

    def calculate_criterion(self, outputs, targets):
        """This function can be overwritten to combine different criterions."""
        loss = self.criterion_fn(outputs, targets)
        return loss

    def get_input_target_output(self, data_batch):
        # Get batch data and place in cuda
        inputs, targets = data_batch
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = self.model.preprocess_input(inputs)
        outputs = self.model.forward(inputs)
        return inputs, targets, outputs

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_model(self, dataloaders, nr_epochs):
        for epoch in range(self.start_epoch, nr_epochs+self.start_epoch):
            print('Epoch {} of {}'.format(epoch, nr_epochs))
            # Set training mode
            self.model.train()
            # Set learning rate
            self.scheduler.step(epoch)
            self.acc.init()
            # Train
            for i, data in enumerate(dataloaders['train']):
                inputs, targets, outputs = self.get_input_target_output(data)
                self.optimizer.zero_grad()
                loss = self.calculate_criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # Track loss
                self.acc.add('loss', loss.item())
            # Save and print epoch statistics
            self.track_statistics(epoch, dataloaders)

            if epoch % self.config.get('model_save_interval', 5) == 0:
                self.save_state(epoch=epoch)

            
