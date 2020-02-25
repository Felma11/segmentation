import torch
assert torch.cuda.is_available()

from src.utils.experiment.Experiment import Experiment
from src.utils.helper_functions import set_gpu
from src.data.datasets import get_dataset
from src.data.data_splitting import split_dataset
from src.data.torcherize import TorchDS

from src.continual_learning.task_splitter import ClassSplitter

def create_experiment(config):
    # Fetch dataset
    dataset = get_class('src.data.datasets.'+config['dataset_name'])(restore=config.get('restore_dataset', True))
    # Create experiment and set indexes for train, val and test
    exp = Experiment(config=config)
    dataset = exp.get_dataset()
    exp.define_runs_splits(dataset)

def run(experiment_name, idx_k=0):
    # Restore configuration and get experiment run
    exp = Experiment(load_exp_name=experiment_name)
    config = exp.config
    exp_run = exp.get_experiment_run(idx_k=idx_k)
    # Fetch dataset and PyTorch dataloaders
    dataset = get_dataset(config)

    task_splitter = ClassSplitter(dataset=dataset, nr_tasks=5)
    datasets = ClassSplitter





    pytorch_datasets = {split: TorchDS(dataset_obj=dataset, index_list=exp_run.dataset_ixs[split]) for split in ['train', 'val', 'test']}
    batch_size = config.get('batch_size', 128)
    dataloaders = dict()
    for split in ['train', 'val', 'test']:
        shuffle = True if split == 'train' else False
        dataloaders[split] = torch.utils.data.DataLoader(pytorch_datasets[split], batch_size=batch_size, shuffle=shuffle)
    print('Data fetched')
    # 


    


config = {
    'cross_validation': True, 
    'nr_runs': 5,
    'val_ratio': 0.2,
    'experiment_name': 'MNIST_trial',
    'dataset_name': 'mnist'}

experiment_name='MNIST_trial'



set_gpu(config.get('gpu', 0))
#create_experiment(config)
run(experiment_name, idx_k=0)