import torch.nn as nn


__all__ = ['toy']


class ToyNet(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        
        super(ToyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, 28, 28]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, 9, 9]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, 3, 3]),
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(3*3*16, 10)
        )

    def forward(self, x, bfgs=False):
        x = self.features(x)
        return x


def toy(**kwargs):
    model = ToyNet(**kwargs)
    return model
