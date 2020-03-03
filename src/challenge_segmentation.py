# %%
from IPython import get_ipython
get_ipython().magic('load_ext autoreload') 
get_ipython().magic('autoreload 2')

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
    'nr_runs': 3,
    'val_ratio': 0.2,
    'experiment_name': 'seg_challenge_experiment',
    'dataset_name': 'segChallengeProstate',
    'model_class_path': '',
    'agent_class_path': '',
    'weights_file_name': ''}

create_experiment(config)
'''
# %%
experiment_name='seg_challenge_experiment'

# Restore configuration and get experiment run
exp = Experiment(load_exp_name=experiment_name)
config = exp.config
exp_run = exp.get_experiment_run(idx_k=0)

# Fetch dataset
dataset = get_dataset(config)

# %%
# Get PyTorch datasets and dataloaders
splits = ['train', 'val', 'test']
batch_size = config.get('batch_size', 20)
pytorch_datasets = {split: None for split in splits}
dataloaders = dict()

for split in splits:
    transform = 'aug' if split == 'train' else 'crop'
    pytorch_datasets[split] = TorchSegmentationDataset(dataset_obj=dataset, 
    index_list=exp_run.dataset_ixs[split], transform=transform)
    shuffle = True if split=='train' else False
    dataloaders[split] = torch.utils.data.DataLoader(pytorch_datasets[split], 
    batch_size=batch_size, shuffle=shuffle)

# %%

from src.eval.visualization.visualize_imgs import plot_overlay_mask
dataloader = dataloaders['train']
for x, y in dataloader:
    for i in range(len(x)):
        img, mask = x[i], y[i]
        if torch.nonzero(mask).size(0) > 0:
            plot_overlay_mask(img, mask)
    break



# %%
model = UNet(n_class=1, n_input_layers=25, n_input_channels=1).cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
results = PartialResult(name='results', metrics=['dice', 'bce', 'loss'])
agent = UNetAgent(config=config, exp_paths=exp_run.paths, model=model, 
    scheduler=exp_lr_scheduler, optimizer=optimizer, results=results, agent_name='aug')

# %%
task_ix = 2
agent.train_model(dataloaders[task_ix], nr_epochs=300)
exp_run.finish(results=results)
# %%
