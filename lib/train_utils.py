from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
from lib.utils import debug
import matplotlib.pyplot as plt

def train_ca(ca, CATrainer, x0, target, optimizer, model_name, 
            scaler=None, global_params=None, training_params=None, model_params=None,
            Visualizer = None, visualize_step = 100, figs_dir = 'figs/'):
    """
    Train a CA model.
    Saves the model and loss history to disk.
    """
    loss_logs = {}
    n_epoch = training_params['n_epoch']
    
    for i in tqdm(range(n_epoch+1)):
        output_dict = CATrainer(ca, x0, target, np.random.randint(global_params['MIN_STEPS'],global_params['MAX_STEPS']), optimizer, scaler, 
                global_params=global_params, training_params=training_params, model_params=model_params)
        
        if i == 0:
            # initialize loss logs
            for key in output_dict.keys():
                if 'loss' in key:
                    loss_logs[key] = []
        
        for key in loss_logs.keys():
            loss_logs[key].append(output_dict[key])
        
        # save loss history to csv file
        if i % 100 == 0:
            df = pd.DataFrame(loss_logs)
            df.to_csv(os.path.join(training_params['history_save_dir'], model_name + '_history.csv'), index=False)

        # visualize training
        if Visualizer is not None and i%visualize_step == 0:
            Visualizer(output_dict, loss_logs, figs_dir+model_name+'.png')
            plt.show()

        # save model
        if i%n_epoch == 0 and i>0:
            torch.save(ca.state_dict(), training_params['model_save_dir'] + model_name +f"_epoch{i}.pth")
            print("Model saved to", training_params['model_save_dir'] + model_name +f"_epoch{i}.pth")