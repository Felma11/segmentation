import torch
import torch.optim as optim
import time
import copy
from collections import defaultdict
import math

from src.models.UNet import UNet
from src.train.losses import calc_loss, calc_losses
from src.utils.load_restore import join_path
from src.utils.accumulator import Accumulator
 
def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def evaluate_model(device, model, dataloaders):
    acc = Accumulator(['ce', 'dice', 'loss'])
    for phase in ['train', 'val']:
        print('Split: '+phase)
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            ce, dice, loss = calc_losses(outputs, labels, bce_weight=0.5)
            nr_samples = labels.size(0)
            acc.add('ce', ce, nr_samples)
            acc.add('dice', dice, nr_samples)
            acc.add('loss', loss, nr_samples)
        for key in ['ce', 'dice', 'loss']:
            print('{}: {}'.format(key, acc.mean(key)))

def train_model(device, storage_path, dataloaders, model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics, bce_weight=0.5)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model,  join_path([storage_path, "best_model.pt"])) #TODO: probably doesnt work, file is to small...

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

