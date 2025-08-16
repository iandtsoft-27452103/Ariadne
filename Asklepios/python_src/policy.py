import torch
import torch.nn as nn
import torch.nn.functional as F
#import label

ch = 256

#biasクラスはdlshogiからの移植
class bias(nn.Module):
    def __init__(self, shape):
        super(bias, self).__init__()
        self.bias=nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        return input + self.bias

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

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
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
        #self.b16 = block()
        #self.b17 = block()
        #self.b18 = block()
        #self.b19 = block()
        #self.b20 = block()
        #self.b21 = block()
        #self.b22 = block()
        #self.b23 = block()
        #self.b24 = block()
        #self.b25 = block()
        #self.b26 = block()
        #self.b27 = block()
        #self.b28 = block()
        #self.b29 = block()
        #self.b30 = block()
        self.l2 = nn.Conv2d(ch, 32, 1, 1, 0, 1, 1, False)
        self.l2_bias = bias(9)

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
        #h = self.b16.forward(h)
        #h = self.b17.forward(h)
        #h = self.b18.forward(h)
        #h = self.b19.forward(h)
        #h = self.b20.forward(h)
        #h = self.b21.forward(h)
        #h = self.b22.forward(h)
        #h = self.b23.forward(h)
        #h = self.b24.forward(h)
        #h = self.b25.forward(h)
        #h = self.b26.forward(h)
        #h = self.b27.forward(h)
        #h = self.b28.forward(h)
        #h = self.b29.forward(h)
        #h = self.b30.forward(h)
        #h = self.l2(h)
        #h = self.l2_bias(h)
        return h # ※l2, l2_biasをコメント化して戻すのを忘れていた。おそらく良くない。学習させ直すと時間がかかるので、一旦このままにしておく。

#my_module = policy()
#my_module.load_state_dict(torch.load("model.pth"))
#sm = torch.jit.script(my_module)
#sm.save("model2.pth")
