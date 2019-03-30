import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hd1 = nn.Linear(4800, 2400)
        self.hd2 = nn.Linear(2400, 1200)
        self.hd3 = nn.Linear(1200, 600)
        self.output = nn.Linear(600, 2)

    def forward(self, x):
        x = F.relu(self.hd1(x))
        x = F.relu(self.hd2(x))
        x = F.relu(self.hd3(x))
        x = self.output(x)
        print('output',x)
        return x