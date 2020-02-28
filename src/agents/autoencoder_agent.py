from src.agents.agent import Agent

class AutoencoderAgent(Agent):
    def __init__(self, config, model, optimizer, criterion):
        super().__init__(config, model, optimizer, criterion)

    def get_input_target_output(self, data_batch):
        """For an autoencoder, the input is the target."""
        inputs, _ = data_batch
        inputs = inputs.cuda()
        inputs = self.model.preprocess_input(inputs)
        outputs = self.model.forward(inputs)
        return inputs, inputs, outputs
