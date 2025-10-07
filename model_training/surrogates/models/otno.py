import torch
import torch.nn as nn

from .fno import FNO
from ..layers.channel_mlp import ChannelMLP as NeuralopMLP
from ..layers.spectral_convolution import SpectralConv


class OTNO(FNO):
    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_channels=4,
            out_channels=1,
            lifting_channels=256,
            projection_channels=256,
            n_layers=4,
            positional_embedding=None,
            use_mlp=False,
            mlp=None,
            non_linearity=torch.nn.functional.gelu,
            norm=None,
            preactivation=False,
            fno_skip="linear",
            mlp_skip="soft-gating",
            separable=False,
            factorization=None,
            rank=1,
            joint_factorization=False,
            fixed_rank_modes=False,
            implementation="factorized",
            decomposition_kwargs=dict(),
            domain_padding=None,
            domain_padding_mode="one-sided",
            fft_norm="forward",
            SpectralConv=SpectralConv,
            **kwargs
    ):        
        super().__init__(
            n_modes = n_modes,
            hidden_channels = hidden_channels,
            in_channels = in_channels,
            out_channels = out_channels,
            lifting_channels = lifting_channels,
            projection_channels = projection_channels,
            n_layers = n_layers,
            positional_embedding=positional_embedding,
            use_channel_mlp = use_mlp,
            channel_mlp_dropout= mlp['dropout'],
            channel_mlp_expansion= mlp['expansion'],
            non_linearity = non_linearity,
            norm = norm,
            preactivation = preactivation,
            fno_skip = fno_skip,
            mlp_skip = mlp_skip,
            separable = separable,
            factorization = factorization,
            rank = rank,
            joint_factorization = joint_factorization,
            fixed_rank_modes = fixed_rank_modes,
            implementation = implementation,
            decomposition_kwargs = decomposition_kwargs,
            domain_padding = domain_padding,
            domain_padding_mode = domain_padding_mode,
            fft_norm = fft_norm,
            SpectralConv = SpectralConv,
            **kwargs
        )

        self.projection = NeuralopMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            non_linearity=non_linearity,
            n_dim=1,
        )

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    # transports: (1, in_channels, n_s_sqrt, n_s_sqrt)
    def forward(self, transports, idx_decoder):
        """TFNO's forward pass"""
        transports = self.lifting(transports)

        if self.domain_padding is not None:
            transports = self.domain_padding.pad(transports)

        for layer_idx in range(self.n_layers):
            transports = self.fno_blocks(transports, layer_idx)

        if self.domain_padding is not None:
            transports = self.domain_padding.unpad(transports) # (1, hidden_channels, n_s_sqrt, n_s_sqrt)

        transports = transports.reshape(self.hidden_channels, -1).permute(1, 0) # (n_s, hidden_channels)

        out = transports[idx_decoder].permute(1,0)

        out = out.unsqueeze(0)
        out = self.projection(out).squeeze(1)
        return out
