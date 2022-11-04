import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import floor
from lib.utils_vis import make_circle_masks


class PonderCAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, 
    img_size=72, hidden_size_evolve=128, hidden_size_ponder=50,
    max_steps=200, training=True):
        super().__init__()
        
        self.training = training
        self.device = device
        self.channel_n = channel_n
        self.img_size = img_size
        self.max_steps = max_steps
        self.hidden_size_evolve = hidden_size_evolve
        self.hidden_size_ponder = hidden_size_ponder
        
        ## Evolve NN ##
        self.fc0 = nn.Linear(channel_n*3, hidden_size_evolve)
        self.fc1 = nn.Linear(hidden_size_evolve, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        ## PonderNet ##
        self.cnn = CNN(n_input=img_size, n_channels=channel_n,
                      kernel_size=3, n_output=hidden_size_ponder)
        self.lambda_layer = nn.Linear(hidden_size_ponder, 1)

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

        # pass through Evolve NN
        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        # stochastic cell updates
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        # kill cells
        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0, eps=1e-6, damage_at_step=-1, damage_location='random', damaged_in_batch=1):
      # NOTE 1: during training this is always gonna run for max_steps
      # since we just want the network to learn the p_n distribution
      # NOTE 2: in current implementation, x(16 chann)-> PonderNet
      
      # extract batch size for QoL
      batch_size = x.shape[0]

      # propagate to get x_1, and h_1
      x = self.update(x, fire_rate, angle)
      h = self.cnn(x)

      # lists to save p_n, x_n
      p_steps = []
      x_steps = []
      lambda_steps = []

      # vectors to save intermediate values
      un_halted_prob = h.new_ones((batch_size,))  # unhalted probability till step n
      halted = h.new_zeros((batch_size,), dtype=torch.long)  # stopping step
      n_steps = h.new_zeros((batch_size,), dtype=torch.int)

      # main loop
      for n in range(1, self.max_steps + 1):
        n_steps += 1-halted.int() #count steps

        # obtain lambda_n
        if n == self.max_steps:
            lambda_n = h.new_ones(batch_size) if batch_size>1 else torch.tensor(1, dtype=torch.float).to(self.device)
        else:
            # pass through pondernet and add a constant to avoid zeros
            lambda_n = torch.sigmoid(self.lambda_layer(h)).squeeze()+eps


        # obtain p_n, +eps to avoid log(0) in KLDiv
        p_n = un_halted_prob * lambda_n + eps
        
        # apply damage
        if n == damage_at_step:
            damage = 1.0-make_circle_masks(damaged_in_batch, self.img_size, self.img_size, damage_location)[..., None]
            damage = torch.from_numpy(damage).to(self.device)
            x[:damaged_in_batch]*=damage

        # append p_n, x_n
        p_steps.append(p_n)
        x_steps.append(x)
        lambda_steps.append(lambda_n)

        # calculate batches to halt
        halt = torch.bernoulli(lambda_n).to(torch.long)*(1-halted)
        halted += halt

        # track unhalted probability and flip coin to halt
        un_halted_prob = un_halted_prob * (1 - lambda_n)

        # propagate to obtain h_n
        x = self.update(x, fire_rate, angle)
        h = self.cnn(x)
        
        # break if we are in inference and all elements have halting_step
        if not self.training and halted.sum() == batch_size:
          break

      return torch.stack(x_steps), torch.stack(p_steps)/torch.sum(torch.stack(p_steps), dim=0), torch.stack(lambda_steps), n_steps
        


class CNN(nn.Module):
    '''
        Simple convolutional neural network to output a hidden state.

        Parameters
        ----------
        n_input : int
            Size of the input image. We assume the image is a square,
            and `n_input` is the size of one side.

        n_channels : int
            Number of channels in the input

        n_ouptut : int
            Size of the output.

        kernel_size : int
            Size of the kernel.
    '''

    def __init__(self, n_input=72, n_channels=48, n_output=50, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        #self.conv2_drop = nn.Dropout2d() ###NOTE: do we need this?

        # calculate size of convolution output
        self.lin_size = floor((floor((n_input - (kernel_size - 1)) / 2) - (kernel_size - 1)) / 2)
        self.fc1 = nn.Linear(self.lin_size ** 2 * 20, n_output)

    def forward(self, x):
        '''forward pass: we need to transpose x from (W,H,C) to (C,H,W)'''
        x = F.relu(F.max_pool2d(self.conv1(x.transpose(1,3)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))#self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        return x