import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    '''
        Computes the weighted average of the given loss across steps according to
        the probability of stopping at each step.
        Parameters
        ----------
        loss_func : callable
            Loss function accepting true and predicted labels. It should output
            a loss item for each element in the input batch.
    '''

    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor):
        '''
            Compute the loss.
            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, of shape `(max_steps, batch_size)`.
            y_pred : torch.Tensor
                Predicted outputs, of shape `(max_steps, batch_size)`.
            y : torch.Tensor
                True targets, of shape `(batch_size)`.
            Returns
            -------
            total_loss : torch.Tensor
                Scalar representing the reconstruction loss.
        '''
        total_loss = p.new_tensor(0.)

        for n in range(p.shape[0]):
            loss = (p[n] * self.loss_func(y_pred[n], y)).mean()
            total_loss = total_loss + loss

        return total_loss



class RegularizationLoss(nn.Module):
    '''
        Computes the KL-divergence between the halting distribution generated
        by the network and a geometric distribution with parameter `lambda_p`.
        Parameters
        ----------
        lambda_p : float
            Parameter determining our prior geometric distribution.
        max_steps : int
            Maximum number of allowed pondering steps.
    '''

    def __init__(self, lambda_p: float, max_steps: int = 1_000, device=None):
        super().__init__()

        p_g = torch.zeros((max_steps,), device=device)
        not_halted = 1.
        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)
        p_g = p_g/torch.sum(p_g, dim=0)
        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.Tensor):
        '''
            Compute the loss.
            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, representing our
                halting distribution.
            Returns
            -------
            loss : torch.Tensor
                Scalar representing the regularization loss.
        '''
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        # normalize p_g
        p_g = p_g.transpose(0,1)/torch.sum(p_g, dim=1)
        p_g = p_g.transpose(0,1)

        return self.kl_div(p.log(), p_g)



class ReconstructionLoss_AdaptFireRate(nn.Module):
    '''
        Computes the weighted average of the given loss across steps according to
        the probability of stopping at each step.
        Parameters
        ----------
        loss_func : callable
            Loss function accepting true and predicted labels. Cellwise loss.
    '''

    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor):
        '''
            Compute the loss.
            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, for each cell, shape `(max_steps, batch_size, img_size, img_size)`.
            y_pred : torch.Tensor
                Predicted outputs, of shape `(max_steps, batch_size, img_size, img_size, channels)`.
            y : torch.Tensor
                True targets, of shape `(batch_size)`.
            Returns
            -------
            total_loss : torch.Tensor
                Scalar representing the reconstruction loss = mean cellwise loss, weighted
                by the halting probabilities
        '''
        # loss as in PonderNet paper: sum of cellwise losses, weighted by p_n
        #total_loss = torch.einsum("abcd,abcd ->bcd", p, self.loss_func(y_pred,y)).mean(dim=[-1,-2])
        
        ### without p_n weighting, only final step counts
        total_loss = self.loss_func(y_pred[-1],y).mean(dim=[-1,-2])
        return total_loss
        
        

class RegularizationLoss_AdaptFireRate(nn.Module):
    '''
        Computes the KL-divergence between the halting distribution generated
        by the network and a geometric distribution with parameter `lambda_p`.
        Parameters
        ----------
        lambda_p : float
            Parameter determining our prior geometric distribution.
        max_steps : int
            Maximum number of allowed pondering steps.
    '''

    def __init__(self, lambda_p: float, max_steps: int = 1000, device=None):
        super().__init__()

        p_g = torch.zeros((max_steps,), device=device)
        not_halted = 1.
        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)
        p_g = p_g/torch.sum(p_g, dim=0)

        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, p: torch.Tensor):
        '''
            Compute the loss.
            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, representing our
                halting distribution.
            Returns
            -------
            loss : torch.Tensor
                Scalar representing the regularization loss.
        '''
        p = p.transpose(0, 1)
        p_g = p.new_ones(p.shape)
        p_g = torch.einsum("b, abcd->abcd", self.p_g, p_g)

        cellwise_div = self.kl_div(p.log(), p_g).sum(dim=1)/p.shape[0]
        return cellwise_div.mean(dim=[-1,-2])