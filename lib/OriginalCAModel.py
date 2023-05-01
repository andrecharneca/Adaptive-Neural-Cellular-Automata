import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.utils_vis import to_rgb
import matplotlib.pyplot as plt

""" CA model with no changes to the original PyTorch code
    = removed grad clipping and tanh activation from CAModel.py"""

class OriginalCAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super().__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        # Gradients in perpendicular directions
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        # concatenate both gradient directions and cell state
        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        # pass through NN
        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate

        # stochastic cell updates
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1]) < fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        # kill cells
        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        update_grid = (stochastic * life_mask.transpose(1,3)).type(torch.bool)
        return x.transpose(1,3), update_grid.squeeze(-1)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0, get_update_grid=False):
        for step in range(steps):
            x, update_grid = self.update(x, fire_rate, angle)
        if get_update_grid:
            return x, update_grid
        else:
            return x

def OriginalCAModelTrainer(ca, x, target, steps, optimizer, scaler = None,scheduler=None,
                global_params=None, training_params=None, model_params=None):
    """
    Trains OriginalCAModel for 1 epoch
    """
    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        with torch.cuda.amp.autocast():
            x = ca(x, steps=steps)
            loss = F.mse_loss(x[:, :, :, :4], target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
    else:
        x = ca(x, steps=steps)
        loss = F.mse_loss(x[:, :, :, :4], target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return {"x": x, "loss": loss.item()}


def OriginalCAModelVisualizer(output_dict, loss_logs, fig_path):
    '''
      View batch final state, and loss
    '''
    x = output_dict['x'].detach().cpu().numpy()

    vis_final = to_rgb(x)

    plt.figure(figsize=[25,12])
    n_cols = x.shape[0]
    n_rows = 2

    # final states
    for i in range(vis_final.shape[0]):
      plt.subplot(n_rows,n_cols,i+1)
      plt.imshow(vis_final[i])
      plt.text(0,0, "final")
      plt.axis('off')
    
    # visualize loss
    plt.subplot(n_rows, 3, 3*n_rows-2)
    plt.plot(np.log10(loss_logs['loss']),  '.', alpha=0.2)
    plt.title('loss')

    # save figure
    plt.savefig(fig_path)
    plt.close()