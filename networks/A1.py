from torch import nn


class NetA1(nn.Module):
    def __init__(self, num_classes):
        super(NetA1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4)
        self.flatten = nn.Flatten(start_dim=-3)
        self.linear1 = nn.Linear(2500, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.softmax(x)
        return x

