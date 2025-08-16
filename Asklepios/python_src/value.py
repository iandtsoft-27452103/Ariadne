import torch
import torch.nn as nn
import torch.nn.functional as F
#import label

ch = 256
fcl = 256

class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1, bias = False)
        self.norm1 = nn.BatchNorm2d(ch, eps = 2e-05)
        self.conv2 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1, bias = False)
        self.norm2 = nn.BatchNorm2d(ch, eps = 2e-05)

    def forward(self, x):
        h1 = F.relu(self.norm1(self.conv1(x)))
        h2 = self.norm2(self.conv2(h1))
        return F.relu(x + h2)

class value(nn.Module):
    def __init__(self):
        super(value, self).__init__()
        self.batch_size = 0
        self.l1_1_1 = nn.Conv2d(in_channels = 63, out_channels = ch, kernel_size = 5, padding = 2, bias = False)
        self.l1_1_2 = nn.Conv2d(in_channels= 63, out_channels = ch, kernel_size = 1, padding = 0, bias = False)
        self.l1_2 = nn.Conv2d(in_channels = 56, out_channels = ch, kernel_size = 1, bias = False) # pieces_in_hand
        self.l1_3 = nn.Conv2d(in_channels = 1, out_channels = ch, kernel_size = 1, bias = False)
        self.n1 = nn.BatchNorm2d(ch, eps = 2e-05)
        self.b1 = block()
        self.b2 = block()
        self.b3 = block()
        self.b4 = block()
        self.b5 = block()
        self.b6 = block()
        self.b7 = block()
        self.b8 = block()
        self.b9 = block()
        self.b10 = block()
        #self.b11 = block()
        #self.b12 = block()
        #self.b13 = block()
        #self.b14 = block()
        #self.b15 = block()
        self.l2 = nn.Conv2d(ch, 32, 1, 1, 0, 1, 1, True)
        self.n2 = nn.BatchNorm2d(32, eps = 2e-05)
        self.v1 = nn.Linear(81 * 32, fcl)
        self.v2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2, x3):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1_3 = self.l1_3(x3)
        u1 = u1_1_1 + u1_1_2 + u1_2 + u1_3
        h = F.relu(self.n1(u1))
        h = self.b1.forward(h)
        h = self.b2.forward(h)
        h = self.b3.forward(h)
        h = self.b4.forward(h)
        h = self.b5.forward(h)
        h = self.b6.forward(h)
        h = self.b7.forward(h)
        h = self.b8.forward(h)
        h = self.b9.forward(h)
        h = self.b10.forward(h)
        #h = self.b11.forward(h)
        #h = self.b12.forward(h)
        #h = self.b13.forward(h)
        #h = self.b14.forward(h)
        #h = self.b15.forward(h)
        h = F.relu(self.n2(self.l2(h)))
        h = h.reshape(self.batch_size, 32, 81)
        h = h.reshape(self.batch_size, 81 * 32)
        h = F.relu(self.v1(h))
        h = self.v2(h)
        h = h.reshape(-1)
        return h

#my_module = value()
#my_module.load_state_dict(torch.load("model_value.pth"))
#sm = torch.jit.script(my_module)
#sm.save("model_value2.pth")
