import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func


def scaled_sigmoid(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Applies a sigmoid function scaled to output values in the range [min_val, max_val].
    This transformation maps any real-valued input to a specified bounded interval,
    maintaining gradient flow for backpropagation. Useful for constraining network outputs.

    Math:
        output = min_val + (max_val - min_val) * σ(x)
        where σ(x) = 1/(1 + exp(-x)) is the standard sigmoid function

    Require:
        max_val >= min_val
    """
    return min_val + (max_val - min_val) * torch.sigmoid(x)


def scaled_logit(y: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Inverse of scaled_sigmoid - maps values from [min_val, max_val] back to unbounded space.

    Also known as the generalized logit transform. Handles numerical stability at boundaries.

    Args:
        y: Input tensor (values must be in (min_val, max_val) range)
        min_val: Lower bound of input range (exclusive)
        max_val: Upper bound of input range (exclusive)

    Returns:
        Tensor of same shape as input with unbounded real values

    Math:
        output = log( (y - min_val) / (max_val - y) )
        This is the inverse operation of scaled_sigmoid()

    Require:
        min_val < y <  max_val
    """
    return torch.log((y - min_val) / (max_val - y))


def compute_Fourier_modes_helper(ndims, nks, Ls):
    '''
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k, cannot have 0

        Parameters:  
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[nmodes, ndims]
    '''
    assert (len(nks) == len(Ls) == ndims)
    if ndims == 1:
        nk, Lx = nks[0], Ls[0]
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nk + 1):
            k_pairs[i, :] = 2 * np.pi / Lx * kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2 * nx * ny + nx + ny
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if (ky == 0 and kx <= 0):
                    continue

                k_pairs[i, :] = 2 * np.pi / Lx * kx, 2 * np.pi / Ly * ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4 * nx * ny * nz + 2 * (nx * ny + nx * nz + ny * nz) + nx + ny + nz
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if (kz == 0 and (ky < 0 or (ky == 0 and kx <= 0))):
                        continue

                    k_pairs[i, :] = 2 * np.pi / Lx * kx, 2 * np.pi / Ly * ky, 2 * np.pi / Lz * kz
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")

    k_pairs = k_pairs[np.argsort(k_pair_mag, kind='stable'), :]
    return k_pairs


def compute_Fourier_modes(ndims, nks, Ls):
    '''
    Compute `nmeasures` sets of Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

        Parameters:  
            ndims : int
            nks   : int[ndims * nmeasures]
            Ls    : float[ndims * nmeasures]

        Return :
            k_pairs : float[nmodes, ndims, nmeasures]
    '''
    assert (len(nks) == len(Ls))
    nmeasures = len(nks) // ndims
    k_pairs = np.stack([compute_Fourier_modes_helper(ndims, nks[i * ndims:(i + 1) * ndims], Ls[i * ndims:(i + 1) * ndims]) for i in range(nmeasures)], axis=-1)

    return k_pairs


def compute_Fourier_bases(nodes, modes):
    '''
    Compute Fourier bases for the whole space
    Fourier bases are cos(kx), sin(kx), 1

        Parameters:  
            nodes        : float[batch_size, nnodes, ndims]
            modes        : float[nmodes, ndims, nmeasures]

        Return :
            bases_c, bases_s : float[batch_size, nnodes, nmodes, nmeasures]
            bases_0 : float[batch_size, nnodes, 1, nmeasures]
    '''
    # temp : float[batch_size, nnodes, nmodes, nmeasures]
    temp = torch.einsum("bxd,kdw->bxkw", nodes, modes)

    bases_c = torch.cos(temp)
    bases_s = torch.sin(temp)
    batch_size, nnodes, _, nmeasures = temp.shape
    bases_0 = torch.ones(batch_size, nnodes, 1, nmeasures, dtype=temp.dtype, device=temp.device)
    return bases_c, bases_s, bases_0
