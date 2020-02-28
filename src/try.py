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

    # Set GPU
    set_gpu(config.get('gpu', 0))
    
    # Fetch dataset
    dataset = get_dataset(config)

    # Create task and class mappings
    nr_tasks = 5
    task_splitter = ClassTaskSplitter(dataset=dataset, save_path=exp_run.paths['obj'], nr_tasks=nr_tasks)

    # Get PyTorch datasets and dataloaders
    splits = ['train', 'val', 'test']
    batch_size = config.get('batch_size', 128)
    pytorch_datasets = [{split: None for split in splits} for task_ix in range(nr_tasks)]
    dataloaders = [{split: None for split in splits} for task_ix in range(nr_tasks)]
    for task_ix in range(nr_tasks):
        for split in splits:
            index_list = task_splitter.get_task_ixs(exp_ixs=exp_run.dataset_ixs[split], task_ix=task_ix)
            pytorch_datasets[task_ix][split] = TorchDS(dataset_obj=dataset, index_list=index_list, transform='pretrained')
            shuffle = True if split == 'train' else False
            dataloaders[task_ix][split] = torch.utils.data.DataLoader(pytorch_datasets[task_ix][split], batch_size=batch_size, shuffle=shuffle)

    # Get model and agent
    model = get_class(config['model_class_path'])(config, in_channels=dataset.nr_channels, img_size=dataset.img_shape)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.MSELoss()
    agent = get_class(config['agent_class_path'])(config, model=model, optimizer=optimizer, criterion=criterion)

    # Train with data from the first task
    for task_ix in range(nr_tasks):
        agent.train_model(dataloaders=dataloaders[task_ix], nr_epochs=20)
        agent.save_state(path=exp_run.paths['agent_states'], name='task_'+str(task_ix))

'''
config = {
    'cross_validation': True, 
    'nr_runs': 5,
    'val_ratio': 0.2,
    'experiment_name': 'MNIST_trial',
    'dataset_name': 'mnist',
    'model_class_path': 'src.models.autoencoding.cnn_autoencoder.CNNAutoencoder',
    'agent_class_path': 'src.agents.autoencoder_agent.AutoencoderAgent',
    'weights_file_name': 'cnn_ae_init'}
create_experiment(config)
'''
experiment_name='MNIST_trial'
run(experiment_name, idx_k=0)
