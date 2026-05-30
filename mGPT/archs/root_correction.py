import torch
import torch.nn as nn


class RootCorrectionHead(nn.Module):
    """Small temporal conv head that predicts residual root channels."""

    def __init__(
        self,
        nfeats: int,
        hidden_dims=(256, 128),
        kernel_size: int = 3,
        output_dim: int = 3,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        dims = [int(nfeats), *[int(dim) for dim in hidden_dims]]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding))
            layers.append(nn.GELU())
        layers.append(nn.Conv1d(dims[-1], int(output_dim), 1))
        self.net = nn.Sequential(*layers)

        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.permute(0, 2, 1)
        delta = self.net(x).permute(0, 2, 1)
        return delta
