from torch import nn


class NetA1(nn.Module):
    def __init__(self, num_classes):
        super(NetA1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)
        self.flatten = nn.Flatten(start_dim=-3)
        self.linear1 = nn.Linear(2304, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def freeze_conv_param(self):
        # Freeze CONV1 parameters
        for param in self.conv1.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.softmax(x)
        return x

