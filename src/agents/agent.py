# ------------------------------------------------------------------------------
# Agents contain models, which they train, as well as additional functionality.
# ------------------------------------------------------------------------------

from src.utils.accumulator import Accumulator
from src.utils.pytorch.pytorch_load_restore import load_model_state, save_model_state, save_optimizer_state, load_optimizer_state

class Agent():
    def __init__(self, config, model, optimizer, criterion):
        self.config = config
        self.model = model
        self.acc = Accumulator()

        # TODO: initialize scheduler, optimizer and criterion based on the config
        #self.scheduler = None
        self.optimizer = optimizer
        self.criterion_fn = criterion

    def save_state(self, path, name):
        """Save all necessary to recuperate the state."""
        save_model_state(self.model, name=name, path=path)
        save_optimizer_state(self.optimizer, name=name, path=path)

    def restore_state(self, path, name):
        restored_model = load_model_state(self.model, name=name, path=path)
        restored_optimizer = load_optimizer_state(self.optimizer, name=name, path=path)
        return restored_model and restored_optimizer

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
        if epoch % self.config.get('tracking_interval', 5) == 0:
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

    def train_model(self, dataloaders, nr_epochs):
        for epoch in range(nr_epochs):
            print('Epoch {} of {}'.format(epoch, nr_epochs))
            # Set training mode
            self.model.train()
            # Set learning rate
            #self.scheduler.step(epoch)
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
