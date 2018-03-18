from capsule_layer import CapsuleLinear
from torch import nn


class CIFAR10CapsuleNet(nn.Module):
    def __init__(self, routing_type='sum', num_iterations=3):
        super(CIFAR10CapsuleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)

        )
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=256, out_capsules=64, in_length=8, out_length=12,
                                                      routing_type='dynamic', share_weight=True,
                                                      num_iterations=num_iterations),
                                        CapsuleLinear(in_capsules=64, out_capsules=10, in_length=12, out_length=16,
                                                      routing_type='contract', share_weight=False,
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 8)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
