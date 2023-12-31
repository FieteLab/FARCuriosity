import torch
from torch import nn
import torch.nn.functional as F

class ActionConvTranspose2d(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 activation='elu'):
        super(ActionConvTranspose2d, self).__init__()

        self.conv = nn.ConvTranspose2d(feature_size + action_size, feature_size, kernel_size, stride=stride, padding=padding)
        self.activation = activation

    def forward(self, x, action):
        h, w = x.shape[-2:]
        action = action.repeat((1,1, h, w))
        x = torch.cat((x, action), 1)
        out = self.conv(x)
        if self.activation == 'elu':
            out = F.elu(out)
        return out

class ActionMemoryConvTranspose2d(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size,
                 memory_size,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 activation='elu'):
        super(ActionMemoryConvTranspose2d, self).__init__()

        self.conv = nn.ConvTranspose2d(feature_size + action_size + memory_size, feature_size, kernel_size, stride=stride, padding=padding)
        self.activation = activation

    def forward(self, x, action, memory):
        h, w = x.shape[-2:]
        action = action.repeat((1,1, h, w))
        memory = F.interpolate(memory, size=(h,w), mode='bilinear')
        x = torch.cat((x, action, memory), 1)
        out = self.conv(x)
        if self.activation == 'elu':
            out = F.elu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel=3, action_size=4, memory_size=10,
            use_action=True, use_memory=False):
        super(Decoder, self).__init__()
        if use_memory:
            self.l1 = ActionMemoryConvTranspose2d(32, action_size, memory_size, 3, stride=2, padding=0)
            self.l2 = ActionMemoryConvTranspose2d(32, action_size, memory_size, 3, stride=2, padding=0)
#            self.l3 = ActionMemoryConvTranspose2d(32, action_size, memory_size, 3, stride=2, padding=1)
            self.l4 = ActionMemoryConvTranspose2d(32, action_size, memory_size, 3, stride=1, padding=1)
        else:
            self.l1 = ActionConvTranspose2d(32, action_size, 3, stride=2, padding=0)
            self.l2 = ActionConvTranspose2d(32, action_size, 3, stride=2, padding=0)
#            self.l3 = ActionConvTranspose2d(32, action_size, 3, stride=2, padding=1)
            self.l4 = ActionConvTranspose2d(32, action_size, 3, stride=1, padding=1)
#        self.l5 = nn.ConvTranspose2d(32, output_channel, 3, stride=1, paadding=0)
        self.l5 = nn.Conv2d(32, output_channel, 3, stride=1, padding=1)
    
    def forward(self, x, action=None, memory=None):
        action = action.unsqueeze(-1).unsqueeze(-1)
        if memory is not None:
            out = self.l1(x, action, memory) 
            out = self.l2(out, action, memory) 
#            out = self.l3(out, action, memory) 
            out = self.l4(out, action, memory)
        else:
            out = self.l1(x, action)
            out = self.l2(out, action)
 #           out = self.l3(out, action) 
            out = self.l4(out, action)
        out = self.l5(out)
        return out

