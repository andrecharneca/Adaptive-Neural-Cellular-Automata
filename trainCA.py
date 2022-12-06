import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from lib.utils import get_free_gpu, load_lizard
from lib.utils_vis import make_seed
from lib.train_utils import train_ca

torch.backends.cudnn.benchmark = True # Speeds up stuff
torch.backends.cudnn.enabled = True

## General Parameters ##
global_params = {
    'CHANNEL_N': 16,
    'TARGET_PADDING': 16,
    'TARGET_SIZE': 40,
    'IMG_SIZE': 72,
    'MIN_STEPS': 64,
    'MAX_STEPS': 96,
}

## General Training Parameters ##
# choose cuda device with the least amount of current memory usage
training_params = {
    'lr': 2e-3,
    'betas': (0.5, 0.5),
    'n_epoch': 16000,
    'batch_size': 8,
    'device': torch.device('cuda:{}'.format(get_free_gpu()) if torch.cuda.is_available() else 'cpu'),
    'history_save_dir': 'histories/',
    'model_save_dir': 'models/',
}

## Model Parameters ##
ca_params = {"CELL_FIRE_RATE": 0.5}

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
from lib.CAModel import CAModel, CAModelTrainer, CAModelVisualizer

# Initialize model
ca = CAModel(global_params["CHANNEL_N"], ca_params['CELL_FIRE_RATE'], training_params["device"])
optimizer = optim.Adam(ca.parameters(), lr=training_params['lr'], betas=training_params['betas'])
model_name = "CA"
scaler = torch.cuda.amp.GradScaler()

## Training ##
print("Currently training:", model_name)
train_ca(ca, CAModelTrainer, 
        x0, pad_target, 
        optimizer, model_name, scaler=scaler,
        global_params=global_params, training_params=training_params, model_params=ca_params,
        Visualizer = CAModelVisualizer, visualize_step=100, figs_dir='figs/')

# run this script with output to a file, and delete the previous output file
# nohup python train.py > train.log 2>&1 &

# delete every file in models/, histories/, and figs/
# rm -rf models/* histories/* figs/*
