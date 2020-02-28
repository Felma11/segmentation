import torch
assert torch.cuda.is_available()
import torch.nn as nn

from src.utils.experiment.Experiment import Experiment
from src.utils.helper_functions import set_gpu
from src.data.datasets import get_dataset
from src.data.data_splitting import split_dataset
from src.data.torcherize import TorchDS

from src.continual_learning.task_splitter import ClassTaskSplitter
from src.utils.introspection import get_class

def create_experiment(config):
    # Fetch dataset
    dataset = get_dataset(config)
    # Create experiment and set indexes for train, val and test
    exp = Experiment(config=config)
    exp.define_runs_splits(dataset)

def run(experiment_name, idx_k=0):
    # Restore configuration and get experiment run
    exp = Experiment(load_exp_name=experiment_name)
    config = exp.config
    exp_run = exp.get_experiment_run(idx_k=idx_k)
    
    # Fetch dataset
    dataset = get_dataset(config)

    # Create task and class mappings
    nr_tasks = 5
    task_splitter = ClassTaskSplitter(dataset=dataset, 
        save_path=exp_run.paths['obj'], nr_tasks=nr_tasks)

    # Get PyTorch datasets
    splits = ['train', 'val', 'test']
    batch_size = config.get('batch_size', 128)
    pytorch_datasets = [{split: None for split in splits} 
        for task_ix in range(nr_tasks)]
    for task_ix in range(nr_tasks):
        for split in splits:
            index_list = task_splitter.get_task_ixs(
                exp_ixs=exp_run.dataset_ixs[split], task_ix=task_ix)
            pytorch_datasets[task_ix][split] = TorchDS(dataset_obj=dataset, 
                index_list=index_list)
    print('Got datasets')

    # Apply oracles
    #task_oracle = get_class('src.continual_learning.oracles.task_oracle.TaskOracle')(
    #    pytorch_datasets=pytorch_datasets, save_path=exp_run.paths['root'])
    
    oracle = get_class('src.continual_learning.oracles.autoencoder_oracle.AutoencoderOracle')(
        pytorch_datasets=pytorch_datasets, save_path=exp_run.paths['root'],
        autoencoder_path='src.models.autoencoding.pretrained_autoencoder.PretrainedAutoencoder',
        feature_model_name='AlexNet')
    print('Initialized oracles')

    # Plot confusion matrix
    for split in splits:
        cm = oracle.get_domain_confusion(split=split)
        cm.plot(exp_run.paths['results'], 
            oracle.name+'_'+split+'_domain_confusion', 
            label_predicted='Selected Model (predicted)', 
            label_actual='Data Task (actual)', figure_size=(7,5))

'''
config = {
    'cross_validation': False, 
    'nr_runs': 1,
    'val_ratio': 0.2,
    'experiment_name': 'MNIST_oracle_trial',
    'dataset_name': 'mnist',
    'model_class_path': '',
    'agent_class_path': '',
    'weights_file_name': ''}
create_experiment(config)
'''
experiment_name='MNIST_oracle_trial'
run(experiment_name, idx_k=0)
#'''