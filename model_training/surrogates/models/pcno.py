import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.pointcloud_convolution import SpectralConv
from ..layers.pointcloud_utils import compute_Fourier_bases, get_act, scaled_logit, scaled_sigmoid


def compute_gradient(f, directed_edges, edge_gradient_weights):
    '''
    Compute gradient of field f at each node
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)   =         :
       :                                :
    xj - x                        f(xj) - f(x)

    in matrix form   dx  nable f(x)   = df.

    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    directed_edges stores directed edges (x, x1), (x, x2), ..., (x, xj)
    edge_gradient_weights stores its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]

    Then the gradient can be computed 
    gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 
    with scatter_add for each edge


        Parameters: 
            f : float[batch_size, in_channels, nnodes]
            directed_edges : int[batch_size, max_nedges, 2] 
            edge_gradient_weights : float[batch_size, max_nedges, ndims]

        Returns:
            x_gradients : float Tensor[batch_size, in_channels*ndims, max_nnodes]
            * in_channels*ndims dimension is gradient[x_1], gradient[x_2], gradient[x_3]......
    '''

    f = f.permute(0, 2, 1)
    batch_size, max_nnodes, in_channels = f.shape
    _, max_nedges, ndims = edge_gradient_weights.shape
    # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
    # target\source : int Tensor[batch_size, max_nedges]
    # message : float Tensor[batch_size, max_nedges, in_channels*ndims]

    target, source = directed_edges[..., 0], directed_edges[..., 1]  # source and target nodes of edges
    message = torch.einsum('bed,bec->becd', edge_gradient_weights, f[torch.arange(batch_size).unsqueeze(1), source] - f[torch.arange(batch_size).unsqueeze(1), target]).reshape(batch_size, max_nedges, in_channels * ndims)

    # f_gradients : float Tensor[batch_size, max_nnodes, in_channels*ndims]
    f_gradients = torch.zeros(batch_size, max_nnodes, in_channels * ndims, dtype=message.dtype, device=message.device)
    f_gradients.scatter_add_(dim=1, src=message, index=target.unsqueeze(2).repeat(1, 1, in_channels * ndims))

    return f_gradients.permute(0, 2, 1)


class PCKNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        inv_L_scale_hyper=['independently', 0.5, 2.0],
        act="gelu",
    ):
        super(PCKNO, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                ndims : int 
                    Dimensionality of the problem
                modes : float[nmodes, ndims, nmeasures]
                    It contains nmodes modes k, and Fourier bases include : cos(k x), sin(k x), 1  
                    * We cannot have both k and -k
                    * k is not integer, and it has the form 2pi*K/L0  (K in Z)
                nmeasures : int
                    Number of measures
                    There might be different integrals with different measures
                layers : list of int
                    number of channels of each layer
                    The lifting layer first lifts to layers[0]
                    The first Fourier layer maps between layers[0] and layers[1]
                    ...
                    The nlayers Fourier layer maps between layers[nlayers-1] and layers[nlayers]
                    The number of Fourier layers is len(layers) - 1
                fc_dim : int 
                    hidden layers for the projection layer, when fc_dim > 0, otherwise there is no hidden layer
                in_dim : int 
                    The number of channels for the input function
                    For example, when the coefficient function and locations (a(x, y), x, y) are inputs, in_dim = 3
                out_dim : int 
                    The number of channels for the output function

                inv_L_scale_hyper: 3 element hyperparameter list
                    Controls the update behavior of the length scale (L) for Fourier modes. The modes are scaled elementwise as:
                    k = k * inv_L_scale (where each spatial direction or measure may be scaled differently).
                    since k = 2pi K /L0, inv_L_scale is the inverse scale, 1/L = inv_L_scale * 1/L0 
                    Hyperparameters: 
                        train_inv_L_scale (bool or str): Update policy for inv_L_scale:
                            False: Disable training (fixed scaling).
                            'together': Train jointly with other parameters (shared optimizer).
                            'independently': Train with a separate optimizer.

                        inv_L_scale_min (float): Lower bound for scaling factor.

                        inv_L_scale_max (float): Upper bound for scaling factor.

                    Implementation Notes:
                        The effective scaling factor is computed via a sigmoid constraint:
                        inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
                        This ensures inv_L_scale stays within [inv_L_scale_min, inv_L_scale_max] during optimization.

                act : string (default gelu)
                    The activation function

            
            Returns:
                Point cloud neural operator

        """

        self.register_buffer('modes', modes)
        self.nmeasures = nmeasures

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max = inv_L_scale_hyper[0], inv_L_scale_hyper[1], inv_L_scale_hyper[2]
        # latent variable for inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
        self.inv_L_scale_latent = nn.Parameter(torch.full((ndims, nmeasures), scaled_logit(torch.tensor(1.0), self.inv_L_scale_min, self.inv_L_scale_max)), requires_grad=bool(self.train_inv_L_scale))

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims * in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = get_act(act)
        self.softsign = F.softsign

        self.normal_params = []  # group of params which will be trained normally
        self.inv_L_scale_params = []  # group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.inv_L_scale_latent:
                self.normal_params.append(param)
            else:
                if self.train_inv_L_scale == 'together':
                    self.normal_params.append(param)
                elif self.train_inv_L_scale == 'independently':
                    self.inv_L_scale_params.append(param)
                elif self.train_inv_L_scale == False:
                    continue
                else:
                    raise ValueError(f"{self.train_inv_L_scale} is not supported")

    def forward(self, x, aux, **kwargs):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters: 
                x : Tensor float[batch_size, max_nnomdes, in_dim] 
                    Input data
                aux : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]  
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]  
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes, nmeasures]  
                                    rho(x)dx used for nmeasures integrations; padding with 0

                    directed_edges : Tensor int[batch_size, max_nedges, 2]  
                                     direted edge pairs; padding with 0  
                                     gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

                    edge_gradient_weights      : Tensor float[batch_size, max_nedges, ndim] 
                                                 pinvdx on each directed edge 
                                                 gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 


            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws)

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux
        # bases: float[batch_size, nnodes, nmodes]
        # scale the modes k  = k * ( inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min)/(1 + exp(-self.inv_L_scale_latent) ))
        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)))
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(self.softsign(compute_gradient(x, directed_edges, edge_gradient_weights)))
            x = x1 + x2 + x3
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x

    def info(self):
        inv_L = scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)
        return tuple(inv_L.flatten().tolist())

    def save_checkpoint(self, save_dir, save_name):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, save_name + '_state_dict.pt'))


class PCFNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        inv_L_scale_hyper=['independently', 0.5, 2.0],
        act="gelu",
    ):
        super(PCFNO, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                ndims : int 
                    Dimensionality of the problem
                modes : float[nmodes, ndims, nmeasures]
                    It contains nmodes modes k, and Fourier bases include : cos(k x), sin(k x), 1  
                    * We cannot have both k and -k
                    * k is not integer, and it has the form 2pi*K/L0  (K in Z)
                nmeasures : int
                    Number of measures
                    There might be different integrals with different measures
                layers : list of int
                    number of channels of each layer
                    The lifting layer first lifts to layers[0]
                    The first Fourier layer maps between layers[0] and layers[1]
                    ...
                    The nlayers Fourier layer maps between layers[nlayers-1] and layers[nlayers]
                    The number of Fourier layers is len(layers) - 1
                fc_dim : int 
                    hidden layers for the projection layer, when fc_dim > 0, otherwise there is no hidden layer
                in_dim : int 
                    The number of channels for the input function
                    For example, when the coefficient function and locations (a(x, y), x, y) are inputs, in_dim = 3
                out_dim : int 
                    The number of channels for the output function

                inv_L_scale_hyper: 3 element hyperparameter list
                    Controls the update behavior of the length scale (L) for Fourier modes. The modes are scaled elementwise as:
                    k = k * inv_L_scale (where each spatial direction or measure may be scaled differently).
                    since k = 2pi K /L0, inv_L_scale is the inverse scale, 1/L = inv_L_scale * 1/L0 
                    Hyperparameters: 
                        train_inv_L_scale (bool or str): Update policy for inv_L_scale:
                            False: Disable training (fixed scaling).
                            'together': Train jointly with other parameters (shared optimizer).
                            'independently': Train with a separate optimizer.

                        inv_L_scale_min (float): Lower bound for scaling factor.

                        inv_L_scale_max (float): Upper bound for scaling factor.

                    Implementation Notes:
                        The effective scaling factor is computed via a sigmoid constraint:
                        inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
                        This ensures inv_L_scale stays within [inv_L_scale_min, inv_L_scale_max] during optimization.

                act : string (default gelu)
                    The activation function

            
            Returns:
                Point cloud neural operator

        """

        self.register_buffer('modes', modes)
        self.nmeasures = nmeasures

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max = inv_L_scale_hyper[0], inv_L_scale_hyper[1], inv_L_scale_hyper[2]
        # latent variable for inv_L_scale = inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min) * sigmoid(inv_L_scale_latent)
        self.inv_L_scale_latent = nn.Parameter(torch.full((ndims, nmeasures), scaled_logit(torch.tensor(1.0), self.inv_L_scale_min, self.inv_L_scale_max)), requires_grad=bool(self.train_inv_L_scale))

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = get_act(act)

        self.normal_params = []  # group of params which will be trained normally
        self.inv_L_scale_params = []  # group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.inv_L_scale_latent:
                self.normal_params.append(param)
            else:
                if self.train_inv_L_scale == 'together':
                    self.normal_params.append(param)
                elif self.train_inv_L_scale == 'independently':
                    self.inv_L_scale_params.append(param)
                elif self.train_inv_L_scale == False:
                    continue
                else:
                    raise ValueError(f"{self.train_inv_L_scale} is not supported")

    def forward(self, x, aux, **kwargs):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           # Note: The differential operator D is deprecated for simplicity
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters: 
                x : Tensor float[batch_size, max_nnomdes, in_dim] 
                    Input data
                aux : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]  
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]  
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes, nmeasures]  
                                    rho(x)dx used for nmeasures integrations; padding with 0

                    directed_edges : Tensor int[batch_size, max_nedges, 2]  
                                     direted edge pairs; padding with 0  
                                     gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

                    edge_gradient_weights      : Tensor float[batch_size, max_nedges, ndim] 
                                                 pinvdx on each directed edge 
                                                 gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 


            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws)

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights = aux
        # bases: float[batch_size, nnodes, nmodes]
        # scale the modes k  = k * ( inv_L_scale_min + (inv_L_scale_max - inv_L_scale_min)/(1 + exp(-self.inv_L_scale_latent) ))
        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)))
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x

    def info(self):
        inv_L = scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)
        return tuple(inv_L.flatten().tolist())

    def save_checkpoint(self, save_dir, save_name):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, save_name + '_state_dict.pt'))
