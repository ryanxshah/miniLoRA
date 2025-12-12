import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, in_features, out_features, rank, alpha, bias=True, device=None, dtype=None):
        super().__init__()

        self.base_linear = base_linear

        # Freeze weights
        self.base_linear.weight.requires_grad_(False)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(False)


        self.lora_a = nn.Linear(in_features=in_features, out_features=rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_features, bias=False, device=device, dtype=dtype)
        
        self.scale_factor = alpha / rank

        nn.init.kaiming_uniform_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base_linear(x) + (self.scale_factor * self.lora_b(self.lora_a(x)))
    

def apply_lora(module: nn.Module, rank, alpha, target_modules):

    for name, childmodule in module.named_children():
        apply_lora(childmodule, rank, alpha, target_modules)

        if isinstance(childmodule, nn.Linear) and name in target_modules:
            setattr(module, name, LoRALinear(childmodule, childmodule.in_features, childmodule.out_features, rank, alpha, childmodule.bias is not None))