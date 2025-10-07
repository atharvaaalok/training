import os
import numpy as np
import torch
import torch.nn as nn


from ..layers.pointcloud_convolution import SpectralConv
from ..layers.pointcloud_utils import compute_Fourier_bases, get_act, scaled_logit, scaled_sigmoid


class BNO(nn.Module):
    def __init__(self,
                 ndims,
                 modes,
                 nmeasures,
                 layers,
                 fc_dim=128,
                 in_dim_u=2,
                 in_dim_v=3,
                 out_dim=1,
                 inv_L_scale_hyper=['independently', 0.5, 2.0],
                 act="gelu"
                 ):
        super(BNO, self).__init__()
        """ 
        A naive implementation of BNO.
        The local operator is replaced by a global one. 
        """

        self.model_type = "BNO"

        self.register_buffer('modes', modes)
        self.nmeasures = nmeasures

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim_u = in_dim_u
        self.in_dim_v = in_dim_v

        self.fc0_u = nn.Linear(in_dim_u, layers[0], fc_dim)
        self.fc0_v = nn.Linear(in_dim_v, layers[0], fc_dim)

        self.sp_convs_ext = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.sp_convs_u = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.sp_convs_v = nn.ModuleList(
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

        self.ws_u = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.ws_v = nn.ModuleList(
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

    def forward(self, u, v, aux, **kwargs):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the boundary neural layers 
                        u' = (W1 + K1)(u) + E(v)
                        v' = (W2 + K2)(u)
           linear functions  W1, W2: parameterized by self.ws_x and self.ws_y; 
           integral operator K1, K2: parameterized by self.sp_convs_x and self.sp_convs_y with nmeasures different integrals
           extension operator E: 
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters: 
                x : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                    Input data in the entire domain
                y : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                    Input data on the boundary
                aux : list of Tensor, containing
                    node_mask_x : Tensor int[batch_size, max_nnomdes_x, 1]  
                    node_mask_y : Tensor int[batch_size, max_nnomdes_y, 1] 
                                1: node; otherwise 0

                    nodes_x : Tensor float[batch_size, max_nnomdes_x, ndim]  
                    nodes_y : Tensor float[batch_size, max_nnomdes_y, ndim] 
                            nodal coordinate; padding with 0

                    node_weights_x  : Tensor float[batch_size, max_nnomdes_x, nmeasures_x]  
                    node_weights_y  : Tensor float[batch_size, max_nnomdes_y, nmeasures_y]  
                                    rho(x)dx used for nmeasures integrations; padding with 0
                                    Currently, we assume nmeasures_x = nmeasures_y = nmeasures = 1 for simplicity.  
                                    The case where x or y has various measures is not yet supported and will be addressed in future updates.                                   

            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws_u)
        node_mask_u, nodes_u, nodes_v, node_weights_u, node_weights_v = aux

        bases_c_u, bases_s_u, bases_0_u = compute_Fourier_bases(nodes_u, self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)))
        bases_c_v, bases_s_v, bases_0_v = compute_Fourier_bases(nodes_v, self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)))

        wbases_c_u = torch.einsum("bxkw,bxw->bxkw", bases_c_u, node_weights_u)
        wbases_s_u = torch.einsum("bxkw,bxw->bxkw", bases_s_u, node_weights_u)
        wbases_0_u = torch.einsum("bxkw,bxw->bxkw", bases_0_u, node_weights_u)

        wbases_c_v = torch.einsum("bxkw,bxw->bxkw", bases_c_v, node_weights_v)
        wbases_s_v = torch.einsum("bxkw,bxw->bxkw", bases_s_v, node_weights_v)
        wbases_0_v = torch.einsum("bxkw,bxw->bxkw", bases_0_v, node_weights_v)

        u = self.fc0_u(u)
        v = self.fc0_v(v)

        u = u.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        for i, (speconv_ext, speconv_u, speconv_v, w_u, w_v) in enumerate(zip(self.sp_convs_ext, self.sp_convs_u, self.sp_convs_v, self.ws_u, self.ws_v)):

            u1 = speconv_ext(v, bases_c_u, bases_s_u, bases_0_u, wbases_c_v, wbases_s_v, wbases_0_v)  # extend operator: boundary to the entire domain
            u2 = speconv_u(u, bases_c_u, bases_s_u, bases_0_u, wbases_c_u, wbases_s_u, wbases_0_u)  # global opertor
            u3 = w_u(u)
            u = u1 + u2 + u3
            if self.act is not None and i != length - 1:
                u = self.act(u)

                # a simple evolution of boundary
                v1 = speconv_v(v, bases_c_v, bases_s_v, bases_0_v, wbases_c_v, wbases_s_v, wbases_0_v)  # this should be a local operator: neighbors of boundary to boundary
                v2 = w_v(v)
                v = v1 + v2
                v = self.act(v)

        u = u.permute(0, 2, 1)

        if self.fc_dim > 0:
            u = self.fc1(u)
            if self.act is not None:
                u = self.act(u)

        u = self.fc2(u)

        return u

    def info(self):
        inv_L = scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)
        return tuple(inv_L.flatten().tolist())

    def save_checkpoint(self, save_dir, save_name):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, save_name + '_state_dict.pt'))


class ExtBNO(nn.Module):
    def __init__(self,
                 ndims,
                 modes,
                 nmeasures,
                 layers,
                 fc_dim=128,
                 in_dim_u=2,
                 in_dim_v=3,
                 out_dim=1,
                 inv_L_scale_hyper=['independently', 0.5, 2.0],
                 act="gelu"
                 ):
        super(ExtBNO, self).__init__()
        """ 
        A naive implementation of ExtBNO.
        The local operator is replaced by a global one. 
        """

        self.model_type = "ExtBNO"

        self.register_buffer('modes', modes)
        self.nmeasures = nmeasures

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim_u = in_dim_u
        self.in_dim_v = in_dim_v

        self.fc0_u = nn.Linear(in_dim_u, layers[0], fc_dim)
        self.fc0_v = nn.Linear(in_dim_v, layers[0], fc_dim)

        self.sp_convs_ext = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.sp_convs_v = nn.ModuleList(
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

        self.ws_u = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.ws_v = nn.ModuleList(
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

    def forward(self, u, v, aux, **kwargs):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the boundary neural layers 
                        u' = W1(u) + E(v)
                        v' = (W2 + K)(u)
           linear functions  W1, W2: parameterized by self.ws_x and self.ws_y; 
           integral operator K: parameterized by self.sp_convs_y with nmeasures different integrals
           extension operator E: 
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters: 
                x : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                    Input data in the entire domain
                y : Tensor float[batch_size, max_nnomdes_x, in_dim] 
                    Input data on the boundary
                aux : list of Tensor, containing
                    node_mask_x : Tensor int[batch_size, max_nnomdes_x, 1]  
                                1: node; otherwise 0

                    nodes_x : Tensor float[batch_size, max_nnomdes_x, ndim]  
                    nodes_y : Tensor float[batch_size, max_nnomdes_y, ndim] 
                            nodal coordinate; padding with 0

                    node_weights_y  : Tensor float[batch_size, max_nnomdes_y, nmeasures_y]  
                                    rho(x)dx used for nmeasures integrations; padding with 0
                                    Currently, we assume nmeasures_x = nmeasures_y = nmeasures = 1 for simplicity.  
                                    The case where x or y has various measures is not yet supported and will be addressed in future updates.                                   

            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """
        length = len(self.ws_u)
        
        node_mask_u, nodes_u, nodes_v, node_weights_v, omega = aux

        inv_L_scale = scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)
        bases_c_u, bases_s_u, bases_0_u = compute_Fourier_bases(nodes_u, self.modes * inv_L_scale)
        bases_c_v, bases_s_v, bases_0_v = compute_Fourier_bases(nodes_v, self.modes * inv_L_scale)

        wbases_c_v = torch.einsum("bxkw,bxw->bxkw", bases_c_v, node_weights_v)
        wbases_s_v = torch.einsum("bxkw,bxw->bxkw", bases_s_v, node_weights_v)
        wbases_0_v = torch.einsum("bxkw,bxw->bxkw", bases_0_v, node_weights_v)

        u = self.fc0_u(u)
        v = self.fc0_v(v)

        u = u.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        for i, (speconv_ext, speconv_v, w_u, w_v) in enumerate(zip(self.sp_convs_ext, self.sp_convs_v, self.ws_u, self.ws_v)):

            u1 = speconv_ext(v, bases_c_u, bases_s_u, bases_0_u, wbases_c_v, wbases_s_v, wbases_0_v)  # extend operator: boundary to the entire domain

            u1 = u1 * torch.prod(inv_L_scale) * omega

            u2 = w_u(u)
            u = u1 + u2
            if self.act is not None and i != length - 1:
                u = self.act(u)

                # a simple evolution of boundary
                v1 = speconv_v(v, bases_c_v, bases_s_v, bases_0_v, wbases_c_v, wbases_s_v, wbases_0_v)  # this should be a local operator: neighbors of boundary to boundary
                v1 = v1 * torch.prod(inv_L_scale) * omega
                
                v2 = w_v(v)
                v = v1 + v2
                v = self.act(v)

        u = u.permute(0, 2, 1)

        if self.fc_dim > 0:
            u = self.fc1(u)
            if self.act is not None:
                u = self.act(u)

        u = self.fc2(u)

        return u

    def info(self):
        inv_L = scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)
        return tuple(inv_L.flatten().tolist())

    def save_checkpoint(self, save_dir, save_name):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, save_name + '_state_dict.pt'))

        
def compute_weights2d(boundary):
    """ 
        Input:      - boundary: (1, m, 2) or (m, 2)
        Return:     - w(x) = m(x) / |Omega|
                    - rho(x) = 1 / |Omega|
    """

    boundary = boundary.squeeze(0)
    left = boundary.roll(1, dims=0)
    right = boundary.roll(-1, dims=0)

    lengths_left = torch.norm(boundary - left, dim=1)
    lengths_right = torch.norm(boundary - right, dim=1)

    weights = 0.5 * (lengths_left + lengths_right)
    weights_sum = weights.sum()

    weights = weights / weights_sum
    rhos = torch.full_like(weights, fill_value=1.0 / weights_sum.item())

    return weights.reshape(1, -1, 1), rhos.reshape(1, -1, 1)


class WrappedExtBNO2D(nn.Module):
    def __init__(self, model, processor):
        super(WrappedExtBNO2D, self).__init__()
        self.model = model
        self.processor = processor
        self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.processor.parameters():
            param.requires_grad = False

    def forward(self, queries, boundary):
        queries = queries.reshape(1, -1, 2)
        boundary = boundary.reshape(1, -1, 2)

        weights, rhos = compute_weights2d(boundary)
        input_dict = dict(
            u=queries,
            v=torch.cat([boundary, rhos], dim=-1),
            aux=(None, queries, boundary, weights)
        )

        out = self.model(**input_dict)

        if self.processor.normalizer is not None:
            out = self.processor.normalizer.inverse_transform(out)

        return out
