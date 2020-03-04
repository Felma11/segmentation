# %%
'''
from IPython import get_ipython
get_ipython().magic('load_ext autoreload') 
get_ipython().magic('autoreload 2')
'''
import torch
assert torch.cuda.is_available()
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.utils.telegram_bot.telegram_bot import TelegramBot
from src.utils.helper_functions import set_gpu

from src.utils.experiment.Experiment import Experiment
from src.utils.helper_functions import set_gpu

from src.data.datasets import get_dataset
from src.data.torcherize import TorchSegmentationDataset
from src.continual_learning.task_splitter import ClassTaskSplitter
from src.eval.results import PartialResult

from src.models.segmentation.UNet import UNet
from src.agents.unet_agent import UNetAgent

# %%
from collections import OrderedDict
import itertools

def define_configs():
    configs = []
    hp = {
        'degrees': [0, 15, 30],
        'translate': [None, (0.2, 0.5), (0.2, 0.5)],
        'scale': [None, (0.8, 1.2), (0.5, 2)],
        'shear': [25]
    }
    hp=OrderedDict(hp.items())
    keys = list(hp.keys())
    combs = list(itertools.product(*hp.values()))
    augs = [{keys[i]: value for (i, value) in enumerate(comb)} for comb in combs]
    for aug in augs:
        config = {
            'gpu': 0,
            'cross_validation': True, 
            'nr_runs': 3,
            'val_ratio': 0.2,
            'experiment_name': 'seg_challenge_experiment',
            'dataset_name': 'segChallengeProstate',
            'lr': 1e-4,
            'lr_decay': (100, 1),
            'nr_epochs': 400,
            'batch_size': 8,
            'aug': aug
            }
        configs.append(config)
    return configs

def create_experiment(config):
    # Fetch dataset
    dataset = get_dataset(config)
    # Create experiment and set indexes for train, val and test
    exp = Experiment(config=config)
    exp.define_runs_splits(dataset)
    return exp, dataset

def run(exp_run, config, dataset, idx_k=0):
    # Fetch dataset
    #dataset = get_dataset(config)

    # Get PyTorch datasets and dataloaders
    splits = ['train', 'val', 'test']
    batch_size = config.get('batch_size', config['batch_size'])
    pytorch_datasets = {split: None for split in splits}
    dataloaders = dict()

    for split in splits:
        transform = 'aug' if split == 'train' else 'crop'
        pytorch_datasets[split] = TorchSegmentationDataset(dataset_obj=dataset, 
        index_list=exp_run.dataset_ixs[split], transform=transform, aug=config['aug'])
        shuffle = True if split=='train' else False
        dataloaders[split] = torch.utils.data.DataLoader(pytorch_datasets[split], 
        batch_size=batch_size, shuffle=shuffle)

    # Get model and agent
    model = UNet(n_class=1, n_input_layers=25, n_input_channels=1).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr']) 
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_decay'][0], gamma=config['lr_decay'][1])
    results = PartialResult(name='results', metrics=['dice', 'bce', 'loss'])
    agent = UNetAgent(config=config, exp_paths=exp_run.paths, model=model, 
        scheduler=exp_lr_scheduler, optimizer=optimizer, results=results, agent_name='aug')

    # Train model
    agent.train_model(dataloaders, nr_epochs=config['nr_epochs'])
    exp_run.finish(results=results)

import sys

if __name__ == '__main__':
    bot = TelegramBot()
    configs = define_configs()[1:]
    gpu = 1
    bot.send_msg('Starting {} configs'.format(len(configs)))
    for ix, config in enumerate(configs):
        print('\nNew experiment {} of {}'.format(ix+1, len(configs)))
        experiment_name = config['experiment_name']+'_'+str(gpu)+'_'+str(ix+1)
        config['experiment_name'] = experiment_name
        exp, dataset = create_experiment(config)
        exp_run = exp.get_experiment_run(idx_k=0)
        try:
            set_gpu(gpu)
            run(exp_run=exp_run, config=config, dataset=dataset)
            state = 'SUCCESS'
        except Exception as e: 
            state = 'ERROR'
            exp_run.finish(exception=e)
        bot.send_msg('Exp. {} of {} PC1 using gpu {} is finished with {}'.format(ix+1, len(configs), gpu, state))


