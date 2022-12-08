import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.utils import debug, plot_loss_reg_rec
from lib.utils_vis import make_circle_masks, damage_batch, to_rgb
import matplotlib.pyplot as plt

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

        # gumbel softmax requires probs for each class, and log probs
        fireRates_gumbel = torch.concat([fireRates, 1-fireRates], dim=-1)
        log_fireRates_gumbel = torch.log(fireRates_gumbel + 1e-10)

        dx = self.fc0(dx)
        dx = F.relu(dx)

        # get fireRates with hidden state
        #fireRates = torch.sigmoid(self.fireRate_layer(dx))

        #debug("fireRates.min()", "fireRates.max()", "dx.max()", "dx.min()")
        dx = self.fc1(dx)

        # stochastic cell updates with gumbel softmax (for differentiable fireRates)
        update_grid = F.gumbel_softmax(log_fireRates_gumbel, tau=1, hard=True, dim=-1)[..., 0].unsqueeze(-1) # 0 is the fireRate class
        dx = dx * update_grid

        x = x+dx.transpose(1,3)

        # kill cells
        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float().to(self.device)
        x = x * life_mask

        return x.transpose(1,3), fireRates.squeeze(-1)

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



def EnergyCAModelTrainer(ca, x, target, steps, optimizer, scaler=None,
                        global_params=None, training_params=None, model_params=None):

    # create decreasing fireRates from MIN->MAX_FIRERATE (right now with fixed steps)
    if model_params['DECAY_TYPE'] == 'Exponential':
        decay_map = torch.from_numpy(np.exp(np.linspace(np.log(model_params['MAX_FIRERATE']),np.log(model_params['MIN_FIRERATE']), global_params['MAX_STEPS']))).to(ca.device)
    elif model_params['DECAY_TYPE'] == 'Linear':
        decay_map = torch.from_numpy(np.linspace(model_params['MAX_FIRERATE'], model_params['MIN_FIRERATE'], global_params['MAX_STEPS']))
    elif model_params['DECAY_TYPE'] == 'None':
        decay_map = torch.ones(global_params['MAX_STEPS']) * model_params['CONST_FIRERATE']

    optimizer.zero_grad(set_to_none=True)

    #with autocast, the dx's become nan for some reason
    x_steps, fireRates_steps = ca(x, steps=steps)
    x_final = x_steps[-1,:,:,:,:4]
    #decay_map = decay_map[0:steps].to(ca.device)
    
    # compute the loss on the live cells
    #goal_fireRate_tensor = torch.einsum("i,ijkl -> ijkl", decay_map, torch.ones(fireRates_steps.shape, device=ca.device))

    # loss computation
    loss_rec_val = F.mse_loss(x_final, target)
    loss_energy_val = torch.tensor(0)#model_params['BETA_ENERGY'] * torch.mean(torch.square(fireRates_steps-goal_fireRate_tensor).sum(dim=[0,2,3]))
    loss = loss_rec_val# + loss_energy_val

    if 0:
          debug(
              "life_mask_steps.shape",
          )
          
    loss.backward()
    optimizer.step()

    return {"x_steps": x_steps, "loss": loss.item(), "loss_rec_val": loss_rec_val.item(), "loss_energy_val": loss_energy_val.item(), "fireRates_steps": fireRates_steps}


def EnergyCAModelVisualizer(output_dict, loss_logs, fig_path, progress_steps = 8):
    '''
      View batch initial seed, batch final state, and progression of fireRates
    of the 1st batch item
    '''
    x = output_dict['x_steps'].detach().cpu().numpy()
    fireRates_steps = output_dict['fireRates_steps'].detach().cpu().numpy()

    vis_final = to_rgb(x[-1])
    vis_batch0 = to_rgb(x[:,0])
    max_steps = x.shape[0]

    plt.figure(figsize=[25,12])
    n_cols = max(x.shape[1], progress_steps)
    n_rows = 4

    # final states
    for i in range(vis_final.shape[0]):
      plt.subplot(n_rows,n_cols,i+1)
      plt.imshow(vis_final[i])
      plt.text(0,0, "n_final="+str(max_steps))
      plt.axis('off')

    # visualize evolution of first in batch
    for i in range(progress_steps):
      plt.subplot(n_rows, n_cols, i+1+n_cols)
      plt.imshow(vis_batch0[max_steps//progress_steps * (i+1)-1])
      plt.text(0,0, "step="+str(max_steps//progress_steps * (i+1)-1))
      plt.axis('off')

    # visualize fireRates
    for i in range(progress_steps):
      plt.subplot(n_rows, n_cols, i+1+2*n_cols)
      plt.imshow(fireRates_steps[max_steps//progress_steps * (i+1)-1,0])
      plt.text(0,0, "fireRates step="+str(max_steps//progress_steps * (i+1)-1))
      plt.axis('off')
      plt.colorbar()
    
    # visualize loss
    plt.subplot(n_rows, 3, 3*n_rows-2)
    plt.plot(np.log10(loss_logs['loss']),  '.', alpha=0.2)
    plt.title('loss')

    plt.subplot(n_rows, 3, 3*n_rows-1)
    plt.plot(np.log10(loss_logs['loss_energy_val']), 'g.', alpha=0.2)
    plt.title('loss_energy_val')

    plt.subplot(n_rows, 3, 3*n_rows)
    plt.plot(np.log10(loss_logs['loss_rec_val']), 'r.', alpha=0.2)
    plt.title('loss_rec_val')

    # save figure
    plt.savefig(fig_path)
