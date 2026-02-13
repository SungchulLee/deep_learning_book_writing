import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(5, 3)
        self.activation = nn.ReLU()
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        return self.head(x)

def freeze_encoder(model: TinyNet):
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            p.requires_grad_(False)

def grad_status(model: nn.Module, title="Grad status"):
    print(title)
    for name, p in model.named_parameters():
        print(f"{name:20s} requires_grad={p.requires_grad}  grad_is_None={p.grad is None}")
    print()

def main():
    model = TinyNet()
    grad_status(model, "Before freezing 'encoder'")
    freeze_encoder(model)
    grad_status(model, "After freezing 'encoder'")

    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    grad_status(model, "After backward")

if __name__ == "__main__":
    main()