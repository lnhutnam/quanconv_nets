import torch
import torch.nn.functional as F
from layers import QuanConv


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quanconv = QuanConv(in_channels=1, out_channels=1, kernel_size=3)
        self.conv = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.dropout = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.quanv(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # x = x.view(-1, 256)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)
