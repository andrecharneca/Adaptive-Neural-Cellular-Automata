import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.utils import debug, plot_loss_reg_rec
from lib.utils_vis import make_circle_masks, damage_batch, to_rgb
import matplotlib.pyplot as plt

class IncrementalEnergyCAModel(nn.Module):
    """ Uses incremental fireRate updates """
    def __init__(self, channel_n, device, hidden_size=128):
        super().__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n*3+1, hidden_size) # +1 for the fireRate
        self.fc0_energyca = nn.Linear(channel_n*3, hidden_size) # used in update_energyca
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        # receives the perception vector and outputs mapping of fireRates
        self.fireRate_layer = nn.Linear(hidden_size, 1)

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

    def update_incremental(self, x, fireRates, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        # pass through NN
        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)

        # concatenate perception vector with fireRate, and compute hidden state
        dx = torch.cat([dx, fireRates.unsqueeze(-1)], dim=-1)
        dx = self.fc0(dx)
        dx = F.relu(dx) # hidden state

        # compute d_inverse_fireRates 
        dinv_fireRates = torch.tanh(self.fireRate_layer(dx))*2 # update vector in the range [-2,2]
        # apply inverse sigmoid funtion to fireRates and sum with dinv_fireRates
        inv_fireRates = - torch.log((1 / (fireRates.unsqueeze(-1) + 1e-8)) - 1) + dinv_fireRates
        fireRates = torch.clamp(torch.sigmoid(inv_fireRates), min=0.1, max=0.9) # min_prob = 0.1, max_prob = 0.9

        # gumbel softmax requires log probs for each class
        fireRates_gumbel = torch.concat([fireRates, 1-fireRates], dim=-1)
        log_fireRates_gumbel = torch.log(fireRates_gumbel)

        dx = torch.tanh(self.fc1(dx)) # update vector in the range [-1,1]

        # stochastic cell updates with gumbel softmax (for differentiable fireRates)
        update_grid = F.gumbel_softmax(log_fireRates_gumbel, tau=0.1, hard=True, dim=-1)[..., 0].unsqueeze(-1).float() # 0 is the fireRate class

        dx = dx * update_grid
        x = x+dx.transpose(1,3)

        # kill cells
        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float().to(self.device)
        x = x * life_mask
        update_grid = (update_grid * life_mask.transpose(1,3)).type(torch.bool)
        return x.transpose(1,3), fireRates.squeeze(-1), update_grid.squeeze(-1)

    def update_energyca(self, x, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        # pass through NN
        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        
        dx = self.fc0_energyca(dx)
        dx = F.relu(dx) #hidden state

        # get fireRates with hidden state
        fireRates = torch.clamp(torch.sigmoid(self.fireRate_layer(dx)), min=1e-7, max=1-1e-7)
        #debug("fireRates.min().item()", "fireRates.max().item()")

        # gumbel softmax requires log probs for each class
        fireRates_gumbel = torch.concat([fireRates, 1-fireRates], dim=-1)
        log_fireRates_gumbel = torch.log(fireRates_gumbel)
        #debug("log_fireRates_gumbel.min().item()", "log_fireRates_gumbel.max().item()")

        dx = torch.tanh(self.fc1(dx)) # update vector in the range [-1,1]
        #debug("dx.min().item()", "dx.max().item()", same_line=True); print()

        # stochastic cell updates with gumbel softmax (for differentiable fireRates)
        update_grid = F.gumbel_softmax(log_fireRates_gumbel, tau=0.1, hard=True, dim=-1)[..., 0].unsqueeze(-1).float() # 0 is the fireRate class
        #update_grid = torch.bernoulli(fireRates)

        dx = dx * update_grid

        x = x+dx.transpose(1,3)

        # kill cells
        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float().to(self.device)
        x = x * life_mask
        update_grid = (update_grid * life_mask.transpose(1,3)).type(torch.bool)
        return x.transpose(1,3), fireRates.squeeze(-1), update_grid.squeeze(-1)

    def forward(self, x, fireRates = None, steps=1, angle=0.0, damage_at_step=-1, damage_location='random', damaged_in_batch=1, get_update_grid=False):
        x_steps = []
        fireRates_steps = []
        update_grid_steps = []
        if fireRates is None:
            fireRates = torch.ones(x.shape[0], x.shape[1], x.shape[2], device=self.device)*0.5

        for step in range(steps):
            # apply damage
            if step == damage_at_step:
                x = damage_batch(x, self.device,img_size = x.shape[-2], damage_location = damage_location, damaged_in_batch = damaged_in_batch)
                fireRates = damage_batch(fireRates.unsqueeze(-1), self.device,img_size = x.shape[-2], damage_location = damage_location, damaged_in_batch = damaged_in_batch)
                fireRates = fireRates.squeeze(-1)

            x, fireRates, update_grid = self.update_energyca(x, angle) 
            x_steps.append(x)
            fireRates_steps.append(fireRates)

            if get_update_grid:
                update_grid_steps.append(update_grid)
        
        if get_update_grid:
            return torch.stack(x_steps), torch.stack(fireRates_steps), torch.stack(update_grid_steps)
        else:
            return torch.stack(x_steps), torch.stack(fireRates_steps)



def IncrementalEnergyCAModelTrainer(ca, x, target, steps, optimizer, scaler=None, scheduler=None,
                        global_params=None, training_params=None, model_params=None):

    optimizer.zero_grad(set_to_none=True)

    #with autocast, the dx's become nan for some reason
    x_steps, fireRates_steps = ca(x, steps=steps)
    x_final = x_steps[-1,:,:,:,:4]

    # loss computation
    loss_rec_val = F.mse_loss(x_final, target)
    ### beta variability for energy loss
    beta = np.random.uniform(model_params['BETA_ENERGY']*0.9, model_params['BETA_ENERGY']*1.1,1)[0]
    loss_energy_val = beta * torch.sum(torch.pow(fireRates_steps, 2))

    loss = loss_energy_val + loss_rec_val

    if model_params['BETA_ENERGY'] == 0:
        # because the plot is in log
        loss_energy_val_for_plot = loss_rec_val.item()
    else:
        loss_energy_val_for_plot = loss_energy_val.item()

    loss.backward()
    # apply gradient clipping
    torch.nn.utils.clip_grad_norm_(ca.parameters(), training_params['grad_clip'])

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    if 0: #debugging
        # inspect model weights and gradients
        for name, param in ca.named_parameters():
            if param.requires_grad:
                print("Param:",name)
                print("\tVal (min, max):",param.data.min(), param.data.max())
                print("\tGrad (min, max):",param.grad.min(), param.grad.max())
            # stop if any weights or grads are NaN   
            if torch.isnan(param.data).any() or torch.isnan(param.grad).any():
                raise ValueError("NaN in weights or grads")


    return {"x_steps": x_steps, "loss": loss.item(), "loss_rec_val": loss_rec_val.item(), "loss_energy_val": loss_energy_val_for_plot, "fireRates_steps": fireRates_steps}


def IncrementalEnergyCAModelVisualizer(output_dict, loss_logs, fig_path, progress_steps = 8):
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
      plt.imshow(fireRates_steps[max_steps//progress_steps * (i+1)-1,0], cmap='jet', vmin=0, vmax=1)
      plt.text(0,0, "fireRates step="+str(max_steps//progress_steps * (i+1)-1))
      plt.axis('off')
      plt.colorbar()
    
    # visualize loss
    # y_range is the min/max of the loss, loss_energy_val, and loss_rec_val
    y_range = [
        min(np.log10(loss_logs['loss']).min(), np.log10(loss_logs['loss_energy_val']).min(), np.log10(loss_logs['loss_rec_val']).min()),
        max(np.log10(loss_logs['loss']).max(), np.log10(loss_logs['loss_energy_val']).max(), np.log10(loss_logs['loss_rec_val']).max())
    ]

    plt.subplot(n_rows, 3, 3*n_rows-2)
    plt.plot(np.log10(loss_logs['loss']),  '.', alpha=0.2)
    plt.title('loss (log)')
    plt.ylim(y_range)

    plt.subplot(n_rows, 3, 3*n_rows-1)
    plt.plot(np.log10(loss_logs['loss_energy_val']), 'g.', alpha=0.2)
    plt.title('loss_energy_val (log)')
    plt.ylim(y_range)

    plt.subplot(n_rows, 3, 3*n_rows)
    plt.plot(np.log10(loss_logs['loss_rec_val']), 'r.', alpha=0.2)
    plt.title('loss_rec_val (log)')
    plt.ylim(y_range)

    # save figure
    plt.savefig(fig_path)
    plt.close()
