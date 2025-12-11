import torch
import torch.nn as nn


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, rank, alpha, bias=True, device=None, dtype=None):

        super().__init__(in_features, out_features, bias, device, dtype)

        self.lora_a = nn.Linear(in_features=in_features, out_features=rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_features, bias=False, device=device, dtype=dtype)

        self.scale_factor = alpha / rank

        nn.init.kaiming_uniform_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)


    def forward(self, x):
        return super().forward(x) + self.scale_factor * self.lora_b(self.lora_a(x))