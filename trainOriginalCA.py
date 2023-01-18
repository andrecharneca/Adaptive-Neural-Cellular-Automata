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
    'MAX_STEPS': 128,#96
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
    'device': torch.device('cuda:{}'.format(get_free_gpu(min_free_mem=0.65)) if torch.cuda.is_available() else 'cpu'),
    'history_save_dir': 'histories/',
    'model_save_dir': f'models/no_gumbel_maxSteps{global_params["MAX_STEPS"]}/', ###
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
from lib.OriginalCAModel import OriginalCAModel, OriginalCAModelTrainer, OriginalCAModelVisualizer

## Training ##
const_fire_rates = [0.278,0.471,0.634,0.689,0.958] 
for const_fire_rate in const_fire_rates:
    ca_params['CELL_FIRE_RATE'] = const_fire_rate

    # Initialize model
    ca = OriginalCAModel(global_params["CHANNEL_N"], ca_params['CELL_FIRE_RATE'], training_params["device"])
    optimizer = optim.Adam(ca.parameters(), lr=training_params['lr'], betas=training_params['betas'])
    model_name = f"OriginalCA_constFireRate{ca_params['CELL_FIRE_RATE']:.2e}_maxSteps{global_params['MAX_STEPS']}"
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=training_params['lr_gamma'])
    scaler = torch.cuda.amp.GradScaler()

    print("Currently training:", model_name)
    
    train_ca(ca, OriginalCAModelTrainer, 
            x0, pad_target, model_name,
            optimizer, scheduler=scheduler,
            global_params=global_params, training_params=training_params, model_params=ca_params,
            Visualizer = OriginalCAModelVisualizer, visualize_step=100, figs_dir='figs/')

# run this script with output to a file, and delete the previous output file
# nohup python trainOriginalCA.py > trainOriginalCA.log 2>&1 &

# delete every file in models/, histories/, and figs/
# rm -rf models/* histories/* figs/*

# look at last line of trainOriginalCA.log
# tail -n 1 trainOriginalCA.log