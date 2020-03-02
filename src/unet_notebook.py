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

from src.utils.experiment.Experiment import Experiment
from src.utils.helper_functions import set_gpu

from src.data.datasets import get_dataset
from src.data.torcherize import TorchSegmentationDataset
from src.continual_learning.task_splitter import ClassTaskSplitter
from src.eval.results import PartialResult

from src.models.segmentation.UNet import UNet
from src.agents.unet_agent import UNetAgent

# %%
'''
def create_experiment(config):
    # Fetch dataset
    dataset = get_dataset(config)
    # Create experiment and set indexes for train, val and test
    exp = Experiment(config=config)
    exp.define_runs_splits(dataset)

config = {
    'cross_validation': True, 
    'nr_runs': 5,
    'val_ratio': 0.2,
    'experiment_name': 'medcom_experiment',
    'dataset_name': 'medcom',
    'dataset_key': 'Manufacturer',
    'model_class_path': '',
    'agent_class_path': '',
    'weights_file_name': ''}

create_experiment(config)
'''
# %%
experiment_name='medcom_experiment'

# Restore configuration and get experiment run
exp = Experiment(load_exp_name=experiment_name)
config = exp.config
exp_run = exp.get_experiment_run(idx_k=0)

# Fetch dataset
dataset = get_dataset(config)

# %%
# Create task and class mappings
nr_tasks = 3
task_splitter = ClassTaskSplitter(dataset=dataset, save_path=exp_run.paths['obj'], nr_tasks=nr_tasks)
print(task_splitter.task_class_mapping)

# %%
# Get PyTorch datasets and dataloaders
splits = ['train', 'val', 'test']
batch_size = config.get('batch_size', 5)
pytorch_datasets = [{split: None for split in splits} for task_ix in range(nr_tasks)]
dataloaders = [{split: None for split in splits} for task_ix in range(nr_tasks)]

for task_ix in range(nr_tasks):
    for split in splits:
        index_list = task_splitter.get_task_ixs(exp_ixs=exp_run.dataset_ixs[split], task_ix=task_ix)
        pytorch_datasets[task_ix][split] = TorchSegmentationDataset(dataset_obj=dataset, index_list=index_list, transform='homogenize')
        shuffle = True if split == 'train' else False
        dataloaders[task_ix][split] = torch.utils.data.DataLoader(pytorch_datasets[task_ix][split], batch_size=batch_size, shuffle=shuffle)

# %%
model = UNet(n_class=1, n_input_layers=25, n_input_channels=1).cuda()

# %%
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)
results = PartialResult(name='results', metrics=['dice', 'bce', 'loss'])
agent = UNetAgent(config=None, model=model, scheduler=exp_lr_scheduler, optimizer=optimizer, results=results)

# %%
task_ix = 2
agent.train_model(dataloaders[task_ix], nr_epochs=500)
exp_run.finish(results=results)
# %%
