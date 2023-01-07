import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from lib.utils import get_free_gpu, load_lizard
from lib.utils_vis import make_seed
from lib.train_utils import train_ca

# For reproducibility
SEED = 1
torch.manual_seed(SEED)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(SEED)

torch.backends.cudnn.benchmark = True # Speeds up stuff
torch.backends.cudnn.enabled = True

## General Parameters ##
global_params = {
    'CHANNEL_N': 16,
    'TARGET_PADDING': 16,
    'TARGET_SIZE': 40,
    'IMG_SIZE': 72,
    'MIN_STEPS': 64,
    'MAX_STEPS': 128,#96,
}

## General Training Parameters ##
# choose cuda device with the least amount of current memory usage
training_params = {
    'lr': 2e-3,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'n_epoch': 10000,
    'batch_size': 8,
    'grad_clip': 1.0,
    'device': torch.device('cuda:{}'.format(get_free_gpu()) if torch.cuda.is_available() else 'cpu'),
    'history_save_dir': 'histories/ignore/',###
    'model_save_dir': 'models/ignore/'#'models/gumbel_maxSteps{}/'.format(global_params["MAX_STEPS"]), ###
}

## Model Parameters ##
energyca_params = {
    "BETA_ENERGY": 1e-7,
}

## Training Initialization ##
# Load target emoji
target_img = load_lizard('data/lizard_clean.png')
p = global_params['TARGET_PADDING']
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(training_params['device'])

# Create initial state
seed = make_seed((h, w), global_params['CHANNEL_N'])
x0 = np.repeat(seed[None, ...], training_params['batch_size'], 0)
x0 = torch.from_numpy(x0.astype(np.float32)).to(training_params['device'])

## Model ##
from lib.EnergyCAModel import EnergyCAModel, EnergyCAModelTrainer, EnergyCAModelVisualizer
    
# grid search on beta_energy
beta_energies = [2e-10]#[0,5e-10, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 1e-8, 1e-7]
for beta_energy in beta_energies:
    energyca_params['BETA_ENERGY'] = beta_energy
    ca = EnergyCAModel(global_params['CHANNEL_N'], training_params['device'])
    optimizer = optim.Adam(ca.parameters(), lr=training_params['lr'], betas=training_params['betas'], weight_decay=1e-5) #some weight decay to avoid exploding weights
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, training_params['lr_gamma'])
    model_name = f"EnergyCA_gumbel_EnergyLoss_betaEnergy{energyca_params['BETA_ENERGY']:.0e}_maxSteps{global_params['MAX_STEPS']}"

    ## Training ##
    print("Currently training:", model_name)
    train_ca(ca, EnergyCAModelTrainer, 
            x0, pad_target, model_name,
            optimizer, scheduler=scheduler,
            global_params=global_params, training_params=training_params, model_params=energyca_params,
            Visualizer = EnergyCAModelVisualizer, visualize_step=100, figs_dir='figs/test/')###

# run this script with output to a file, and delete the previous output file
# nohup python trainEnergyCA.py > trainEnergyCA.log 2>&1 &

# delete every file in models/, histories/, and figs/
# rm -rf models/* histories/* figs/*

# print last 10 lines of a file
# tail -n 10 trainEnergyCA.log

