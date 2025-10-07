import torch
import torch.nn as nn


class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.modes = modes
        self.nmeasures = nmeasures
        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, 1, nmeasures, dtype=torch.float
            )
        )

    def forward(self, x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0):
        '''
        Compute Fourier neural layer
            Parameters:  
                x                   : float[batch_size, in_channels, nnodes]
                bases_c, bases_s    : float[batch_size, nnodes, nmodes, nmeasures]
                bases_0             : float[batch_size, nnodes, 1, nmeasures]
                wbases_c, wbases_s  : float[batch_size, nnodes, nmodes, nmeasures]
                wbases_0            : float[batch_size, nnodes, 1, nmeasures]

            Return :
                x                   : float[batch_size, out_channels, nnodes]
        '''
        x_c_hat = torch.einsum("bix,bxkw->bikw", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s)
        x_0_hat = torch.einsum("bix,bxkw->bikw", x, wbases_0)

        weights_c, weights_s, weights_0 = self.weights_c, self.weights_s, self.weights_0

        f_c_hat = torch.einsum("bikw,iokw->bokw", x_c_hat, weights_c) - torch.einsum("bikw,iokw->bokw", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bikw,iokw->bokw", x_s_hat, weights_c) + torch.einsum("bikw,iokw->bokw", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bikw,iokw->bokw", x_0_hat, weights_0)

        x = torch.einsum("bokw,bxkw->box", f_0_hat, bases_0) + 2 * torch.einsum("bokw,bxkw->box", f_c_hat, bases_c) - 2 * torch.einsum("bokw,bxkw->box", f_s_hat, bases_s)

        return x
