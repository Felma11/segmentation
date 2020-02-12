# ------------------------------------------------------------------------------
# Agents contain a model, which they train, as well as additional functionality.
# ------------------------------------------------------------------------------

class Agent():
    def __init__(self, agent_config, model, dataloaders):
        self.agent_config = agent_config
        self.model = model
        self.dataloaders = dataloaders

        # TODO: initialize scheduler, optimizer and criterion based on the agent_config
        self.scheduler = None
        self.optimizer = None
        self.criterion_fn = None

    def track_statistics(epoch, running_loss):
        # TODO: store running_loss

        # TODO track and print other measures every so many epochs
        if epoch % self.agent_config['tracking_interval'] == 0:
            print()

    def calculate_criterion(outputs, targets):
        """This function can be overwritten to combine different criterions."""
        loss = self.criterion_fn(outputs, targets)
        return out, loss

    def train_model(nr_epochs, optimizer):
        for epoch in range(nr_epochs):
            # Set training mode
            self.model.train()
            # Set learning rate
            self.scheduler.step(epoch)
            
            # Train
            running_loss = 0.0
            for i, data in enumerate(self.dataloaders['train'], 0):

                # Get batch data and place in cuda
                inputs, targets = data
                inputs, targets = inputs.cuda(), targets.cuda()

                # Training step
                self.optimizer.zero_grad()
                outputs = self.model.forward(inputs)
                loss = calculate_criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # Track loss
                running_loss += loss.item()
            
            # Save and print epoch statistics
            track_statistics(epoch, running_loss)