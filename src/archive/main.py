import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from src.models.UNet import UNet
from src.utils.load_restore import join_path, pkl_dump
from src.data.data_processing import load_data
from src.data.data_loader import setup_dataloaders
from src.train.train import train_model, evaluate_model

import torch.nn.functional as F

###define paths etc. ...
raw_data_dir = "C:\\Users\\cgonzale\\Documents\\data\\MedCom_resegmented"
storage_dir = "storage"
vol_size = 512,512,24
vol_spacing = 1,1,3.3 #TODO: Need median spacing for all 3 dims

#TODO: check data type before dump to sace space
def preprocessing():
    x,y = load_data(raw_data_dir, vol_size, vol_spacing)
    pkl_dump([x,y], "data_dump", storage_dir)

def train(device):
    dataloaders = setup_dataloaders(storage_dir, batch_size=16)

    model = UNet(n_class=1, n_input_layers=25, n_input_channels=1).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) 
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = train_model(device, storage_dir, dataloaders, model, optimizer_ft, exp_lr_scheduler, num_epochs=500)
    
    return model

def eval_model(device):
    dataloaders = setup_dataloaders(storage_dir, batch_size=10)
    model = UNet(n_class=1, n_input_layers=25, n_input_channels=1).to(device)
    full_path = join_path([storage_dir, "best_model.pt"])
    state_dict = torch.load(full_path).state_dict()
    model.load_state_dict(state_dict)
    evaluate_model(device, model, dataloaders)

#TODO plot/eval
""" def eval(model, device):
    model.eval()   # Set model to the evaluation mode

    # Create another simulation dataset for test
    test_dataset = SimDataset(3, transform = trans)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

"""
#preprocessing()
#assert torch.cuda.is_available()
#device = torch.device("cuda:0")
#train(device)
#eval_model(device)

#%%
#x,y = load_data(raw_data_dir, vol_size, vol_spacing)
