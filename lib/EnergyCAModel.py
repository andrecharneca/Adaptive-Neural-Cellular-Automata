import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.utils import debug
from lib.utils_vis import make_circle_masks, damage_batch


class EnergyCAModel(nn.Module):
    def __init__(self, channel_n, device, hidden_size=128):
        super().__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        # receives the perception vector and outputs mapping of fireRates
        #self.fireRate_layer = nn.Linear(hidden_size, 1)# do this if input=hidden state
        self.fireRate_layer = nn.Linear(channel_n*3, 1, bias=False) #else the input is the perception vector
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fireRate_layer.weight.zero_()

        self.to(self.device)

    def alive(self, x):        
        ''' If all cells in neighbourhood have alpha<0.1, then
        cell is dead: all channels=0 '''
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

    def update(self, x, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        # pass through NN
        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        # get fireRates with perception vector, fireRates of dead cells is = 0
        fireRates = torch.sigmoid(self.fireRate_layer(dx)) * pre_life_mask.transpose(1,3)

        dx = self.fc0(dx)
        dx = F.relu(dx)

        # get fireRates with hidden state
        #fireRates = torch.sigmoid(self.fireRate_layer(dx))

        #debug("fireRates.min()", "fireRates.max()", "dx.max()", "dx.min()")
        dx = self.fc1(dx)

        # stochastic cell updates
        update_grid = torch.bernoulli(fireRates).float().to(self.device)
        dx = dx * update_grid

        x = x+dx.transpose(1,3)

        # kill cells
        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float().to(self.device)
        x = x * life_mask

        return x.transpose(1,3), fireRates

    def forward(self, x, steps=1, angle=0.0, damage_at_step=-1, damage_location='random', damaged_in_batch=1):
        x_steps = []
        fireRates_steps = []
        for step in range(steps):
            # apply damage
            if step == damage_at_step:
                x = damage_batch(x, self.device,img_size = 72, damage_location = damage_location, damaged_in_batch = damaged_in_batch   )

            x, fireRates = self.update(x, angle)                
            x_steps.append(x)
            fireRates_steps.append(fireRates)
            
        return torch.stack(x_steps), torch.stack(fireRates_steps)
