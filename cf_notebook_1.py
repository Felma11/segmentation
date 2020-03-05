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
from collections import OrderedDict
import itertools

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
def define_configs():
    configs = []
    hp = {
        'degrees': [0],'translate': [(0.2, 0.5)],'scale': [(0.5, 2)],'shear': [5],
        'brightness': [0],'contrast': [0],'saturation': [0],
        'hue': [0]
    }
    hp=OrderedDict(hp.items())
    keys = list(hp.keys())
    combs = list(itertools.product(*hp.values()))
    augs = [{keys[i]: value for (i, value) in enumerate(comb)} for comb in combs]
    for aug in augs:
        config = {
            'gpu': 1,
            'cross_validation': True, 
            'nr_runs': 3,
            'val_ratio': 0.2,
            'experiment_name': 'medcom_experiment',
            'dataset_name': 'medcom',
            'dataset_key': 'Manufacturer',
            'weights_file_path': 'C:\\Users\\cgonzale\\Desktop\\segmentation\\storage\\experiments\\seg_challenge_experiment\\0\\agent_states\\UNetAgent_aug_300_model',
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
    # Create task and class mappings
    nr_tasks = 3
    task_splitter = ClassTaskSplitter(dataset=dataset, save_path=exp_run.paths['obj'], nr_tasks=nr_tasks)
    print(task_splitter.task_class_mapping)

    # Get PyTorch datasets and dataloaders
    splits = ['train', 'val', 'test']
    batch_size = config.get('batch_size', 8)
    pytorch_datasets = [{split: None for split in splits} for task_ix in range(nr_tasks)]
    dataloaders = [{split: None for split in splits} for task_ix in range(nr_tasks)]
    for task_ix in range(nr_tasks):
        for split in splits:
            transform = 'aug' if split == 'train' else 'resize'
            index_list = task_splitter.get_task_ixs(exp_ixs=exp_run.dataset_ixs[split], task_ix=task_ix)
            pytorch_datasets[task_ix][split] = TorchSegmentationDataset(dataset_obj=dataset, 
                index_list=index_list, transform=transform, aug=config['aug'])
            shuffle = True if split == 'train' else False
            dataloaders[task_ix][split] = torch.utils.data.DataLoader(pytorch_datasets[task_ix][split], 
            batch_size=batch_size, shuffle=shuffle)

    # Get model and agent
    results = []
    #for i in range(nr_tasks):
    for i in [2]:
        print('Task {}'.format(i))
        model = UNet(n_class=1, n_input_layers=25, n_input_channels=1, weights_file_path=config.get('weights_file_path', None)).cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr']) 
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_decay'][0], gamma=config['lr_decay'][1])
        task_results = PartialResult(name='results_'+str(i), metrics=['dice', 'bce', 'loss'])
        agent = UNetAgent(config=config, exp_paths=exp_run.paths, model=model, 
            scheduler=exp_lr_scheduler, optimizer=optimizer, results=task_results, agent_name='agentTask'+str(i))
        # Train model
        agent.train_model(dataloaders[i], nr_epochs=config['nr_epochs'])
        results.append(task_results)

    exp_run.finish(results=results)

if __name__ == '__main__':
    bot = TelegramBot()
    configs = define_configs()
    bot.send_msg('Starting {} configs'.format(len(configs)))
    for ix, config in enumerate(configs):
        print('\nNew experiment {} of {}'.format(ix+1, len(configs)))
        gpu = config['gpu']
        experiment_name = config['experiment_name']+'_'+str(gpu)+'_'+str(ix+1)
        config['experiment_name'] = experiment_name

        exp, dataset = create_experiment(config)
        exp_run = exp.get_experiment_run(idx_k=0)
        
        #try:
        set_gpu(gpu)
        run(exp_run=exp_run, config=config, dataset=dataset)
        #    state = 'SUCCESS'
        #except Exception as e: 
        #    state = 'ERROR'
        #    exp_run.finish(exception=e)
        #bot.send_msg('Exp. {} of {} PC1 using gpu {} is finished with {}'.format(ix+1, len(configs), gpu, state))

