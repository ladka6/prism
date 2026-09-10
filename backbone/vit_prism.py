import torch
import torch.nn as nn
import torch.nn.init as init


class PrismAdapter(nn.Module):
    def __init__(self, dim, mlp_ratio=16, se_ratio=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // mlp_ratio),
            nn.GELU(),
            nn.Linear(dim // mlp_ratio, dim),
        )

        self.se = nn.Sequential(
            nn.Linear(dim, dim // se_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(dim // se_ratio, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mlp_out = self.norm(x + self.mlp(x))
        return mlp_out * self.se(mlp_out)


class PrismViT(nn.Module):
    def __init__(self, model, mlp_ratio, se_ratio):
        super().__init__()
        self.vit = model
        self.embed_dim = self.vit.embed_dim

        for param in self.vit.parameters():
            param.requires_grad = False

        self.prism_adapter = PrismAdapter(self.embed_dim, mlp_ratio, se_ratio)

    def forward_features(self, x):
        x = self.vit.forward_features(x)
        if x.dim() == 2:
            return x
        x = self.vit.pool(x)
        x = self.vit.fc_norm(x)
        return x

    def forward_adapter(self, x):
        return self.prism_adapter(x)

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.forward_adapter(x)
        return x

    def reset_adapter(self):
        for m in self.prism_adapter.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)
