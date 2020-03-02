import torch.nn.functional as F

from src.utils.accumulator import Accumulator
from src.agents.agent import Agent
from src.eval.metrics import dice

class UNetAgent(Agent):
    def __init__(self, config, model, scheduler, optimizer):
        super().__init__(config, model, scheduler, optimizer, criterion=None)

    def save_state(self, path, name):
        state_dict = copy.deepcopy(self.model.state_dict())
        save_model_state(state_dict, name=name+'_state_dict', path=path)
        save_model_state(self.model, name=name, path=path)
        save_optimizer_state(self.optimizer, name=name, path=path)

    def restore_state(self, path, name):
        restored_model = load_model_state(self.model, name=name, path=path)
        restored_optimizer = load_optimizer_state(self.optimizer, name=name, path=path)
        return restored_model and restored_optimizer

    def calculate_criterion(self, outputs, targets):
        bce_weight=0.1
        bce = F.binary_cross_entropy_with_logits(outputs, targets)
        pred = F.sigmoid(outputs)
        dice_loss = dice(outputs, targets)
        loss = bce * bce_weight + dice_loss * (1 - bce_weight)
        return loss

    def track_statistics(self, epoch, dataloaders):
        bce_weight=0.1
        if epoch % 5 == 0:
            print('Epoch {}, LR {}'.format(epoch, self.get_lr()))
            self.model.train(False)
            for split in ['train', 'val', 'test']:
                print('Phase: {}'.format(split))
                acc = Accumulator(keys=['dice', 'bce', 'loss'])
                for i, data in enumerate(dataloaders[split]):
                    inputs, targets, outputs = self.get_input_target_output(data)
                    bce = F.binary_cross_entropy_with_logits(outputs, targets)
                    pred = F.sigmoid(outputs)
                    dice_loss = dice(outputs, targets)
                    loss = bce * bce_weight + dice_loss * (1 - bce_weight)
                    acc.add('dice', float(dice_loss), count=len(inputs))
                    acc.add('bce', float(bce), count=len(inputs))
                    acc.add('loss', float(loss), count=len(inputs))
                print('Dice: {}, BCE: {}, Loss: {}'.format(acc.mean('dice')
                , acc.mean('bce'), acc.mean('loss')))