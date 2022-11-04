import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import floor
from lib.utils_vis import make_circle_masks
from lib.utils import debug

    
class AdaptiveFireRateCAModel(nn.Module):
    def __init__(self, channel_n, device, 
    img_size=72, hidden_size_evolve=128, hidden_size_ponder=16,
    max_steps=200, training=True, lambda_eps=1e-2, p_eps=1e-6):
        super().__init__()
        
        self.training = training
        self.device = device
        self.channel_n = channel_n
        self.img_size = img_size
        self.max_steps = max_steps
        self.hidden_size_evolve = hidden_size_evolve
        self.hidden_size_ponder = hidden_size_ponder
        self.lambda_eps = lambda_eps # to clamp lambdas
        self.p_eps = p_eps # to add to p_n
        
        ## Evolve NN ##
        self.fc0 = nn.Linear(channel_n*3, hidden_size_evolve)
        self.fc1 = nn.Linear(hidden_size_evolve, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        ## PonderNet for Adaptive Fire Rate; input: perception vector##
        self.lambda_net = Lambda_CNN(n_input=img_size, n_channels=channel_n*3,
                                kernel_size=3, hidden_size=hidden_size_ponder, eps=lambda_eps)

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

    def update(self, x, lambdas, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        # pass through Evolve NN
        dx = self.perceive(x, angle)
        perc_vec = dx # save for pondernet
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        # stochastic cell updates with lambdas
        # NOTE: this is technically not a Bernoulli process anymore, since cells can halt
        # one step and then update the next step. It's just a different fire_rate for each cell
        update_grid = 1-torch.bernoulli(lambdas).float().to(self.device)
        dx = dx * update_grid.unsqueeze(dim=-1)
        x = x+dx.transpose(1,3)

        # kill cells
        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3), update_grid, perc_vec

    def forward(self, x, angle=0.0, damage_at_step=-1, damage_location='random', damaged_in_batch=1):
      # NOTE 1: during training this is always gonna run for max_steps
      # since we just want the network to learn the p_n distribution
      # NOTE 2: in current implementation, x(46 chann)-> PonderNet
      
      # extract batch size for QoL
      batch_size = x.shape[0]

      # propagate to get x_1
      perc_vec = self.perceive(x.transpose(1,3), angle)
      #max_neighbour_alpha = F.max_pool2d(x[:, :, :, 3], kernel_size=3, stride=1, padding=1)
      
      # from ]-1,1[, not the halting probabilty anymore
      lambdas_n = self.lambda_net(perc_vec) 
      
      # halting probabilities for this step
      #halting_prob_grid = torch.clamp(lambdas_n + (1-max_neighbour_alpha), min=self.lambda_eps, max=1-self.lambda_eps)
      x, update_grid, perc_vec = self.update(x, lambdas_n, angle)

      # lists to save p_n, x_n
      p_steps = []
      x_steps = []
      lambdas_steps = []
      update_grid_steps = []

      # vectors to save intermediate values
      un_halted_prob = x.new_ones((batch_size, self.img_size, self.img_size))  # unhalted probability till step n

      # main loop
      for n in range(1, self.max_steps + 1):
        # obtain lambda_n
        # pass through pondernet
        lambdas_n = self.lambda_net(perc_vec)
        #max_neighbour_alpha = F.max_pool2d(x[:, :, :, 3], kernel_size=3, stride=1, padding=1)
        #halting_prob_grid = torch.clamp(lambdas_n + (1-max_neighbour_alpha), min=self.lambda_eps, max=1-self.lambda_eps)
        #debug("halting_prob_grid.mean()","halting_prob_grid.max()","halting_prob_grid.min()" )
        
        # obtain p_n, +eps to avoid log(0) in KLDiv
        p_n = un_halted_prob * lambdas_n + self.p_eps
        
        # apply damage
        if n == damage_at_step:
            damage = 1.0-make_circle_masks(damaged_in_batch, self.img_size, self.img_size, damage_location)[..., None]
            damage = torch.from_numpy(damage).to(self.device)
            x[:damaged_in_batch]*=damage

        # append tensors
        p_steps.append(p_n)
        x_steps.append(x)
        lambdas_steps.append(lambdas_n)
        update_grid_steps.append(update_grid)

        # track unhalted probability and flip coin to halt
        un_halted_prob = un_halted_prob * (1 - lambdas_n)

        # propagate to obtain h_n
        x, update_grid, perc_vec = self.update(x, lambdas_n, angle)

      return torch.stack(x_steps), torch.stack(p_steps)/torch.sum(torch.stack(p_steps), dim=0), torch.stack(lambdas_steps), torch.stack(update_grid_steps)
        


class Lambda_CNN(nn.Module):
    '''
        Simple convolutional neural network to output the lambdas for each cell.
        Output = Lambdas(batch_size,n_input,n_input

        Parameters
        ----------
        n_input : int
            Size of the input image. We assume the image is a square,
            and `n_input` is the size of one side.

        n_channels : int
            Number of channels in the input

        hidden_size : int
            Number of hidden channels.

        kernel_size : int
            Size of the kernel.
    '''

    def __init__(self, n_input=72, n_channels=48, hidden_size=16, kernel_size=3, eps=1e-1):
        super(Lambda_CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, hidden_size, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv2d(hidden_size, 1, kernel_size=kernel_size, padding='same')
        self.eps = eps
        

    def forward(self, x):
        # the x,y axes come swapped in dx 
        lambdas = F.relu(self.conv1(x.transpose(3,2)))
        lambdas = torch.sigmoid(self.conv2(lambdas))
        
        return lambdas.squeeze(dim=1)