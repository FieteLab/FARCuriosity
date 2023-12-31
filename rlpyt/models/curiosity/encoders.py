
import torch
from torch import nn
import torch.nn.functional as F

from rlpyt.models.utils import Flatten


class ActionConv2d(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size,
                 out_size=None,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 ):
        super(ActionConv2d, self).__init__()
        if out_size is None:
            out_size = feature_size 
        self.conv = nn.Conv2d(feature_size + action_size, out_size, kernel_size, stride=stride, padding=padding)

    def forward(self, x, action):
        h, w = x.shape[-2:]
        action = action.repeat((1,1, h, w))
        x = torch.cat((x, action), 1)
        out = self.conv(x)
        return out



class UniverseHead(nn.Module):
    '''
    Universe agent example: https://github.com/openai/universe-starter-agent
    '''
    def __init__(
            self, 
            image_shape,
            batch_norm=False
            ):
        super(UniverseHead, self).__init__()
        c, h, w = image_shape
        sequence = list()
        for l in range(5):
            if l == 0:
                conv = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            else:
                conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            block = [conv, nn.ELU()]
            if batch_norm:
                block.append(nn.BatchNorm2d(32))
            sequence.extend(block)
        self.model = nn.Sequential(*sequence)


    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state.view(encoded_state.shape[0], -1)

class MacActionHead(nn.Module):
    '''
    World discovery models paper
    '''
    def __init__(
            self, 
            image_shape,
            output_size=256,
            conv_output_size=256,
            batch_norm=False,
            action_size=4,
            ):
        super(MacActionHead, self).__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.conv_output_size = conv_output_size
        self.conv1 = ActionConv2d(c, action_size, 32, (3,3), (1,1), (1,1))
        self.conv2 = ActionConv2d(32, action_size, 32, (3,3), (2,2), (2,2))
        self.conv3 = ActionConv2d(32, action_size, 32, (3,3), (2,2), (2,2))


    def forward(self, state, action):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        state = F.relu(self.conv1(state, action))
        state = F.relu(self.conv2(state, action))
        state = F.relu(self.conv3(state, action))
        return state



class MacHead(nn.Module):
    '''
    World discovery models paper
    '''
    def __init__(
            self, 
            image_shape,
            output_size=256,
            conv_output_size=256,
            batch_norm=False,
            ):
        super(MacHead, self).__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.conv_output_size = conv_output_size
        self.model = nn.Sequential(
                                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                                nn.ReLU(),
                                )

    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state



class MazeHead(nn.Module):
    '''
    World discovery models paper
    '''
    def __init__(
            self, 
            image_shape,
            output_size=256,
            conv_output_size=256,
            batch_norm=False,
            ):
        super(MazeHead, self).__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.conv_output_size = conv_output_size
        self.model = nn.Sequential(
                                nn.Conv2d(in_channels=c, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                                nn.ReLU(),
                                Flatten(),
                                nn.Linear(in_features=self.conv_output_size, out_features=self.output_size),
                                nn.ReLU()
                                )

    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state

class BurdaHead(nn.Module):
    '''
    Large scale curiosity paper
    '''
    def __init__(
            self, 
            image_shape,
            output_size=512,
            conv_output_size=3136,
            batch_norm=False,
            ):
        super(BurdaHead, self).__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.conv_output_size = conv_output_size
        sequence = list()
        sequence += [nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(8, 8), stride=(4, 4)), 
                     nn.LeakyReLU()]
        if batch_norm:
            sequence.append(nn.BatchNorm2d(32))
        sequence += [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)), 
                     nn.LeakyReLU()]
        if batch_norm:
            sequence.append(nn.BatchNorm2d(64))
        sequence += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
                     nn.LeakyReLU()]
        if batch_norm:
            sequence.append(nn.BatchNorm2d(64))
        sequence.append(Flatten())
        sequence.append(nn.Linear(in_features=self.conv_output_size, out_features=self.output_size))

        self.model = nn.Sequential(*sequence)
        # self.model = nn.Sequential(
        #                         nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
        #                         nn.LeakyReLU(),
        #                         # nn.BatchNorm2d(32),
        #                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
        #                         nn.LeakyReLU(),
        #                         # nn.BatchNorm2d(64),
        #                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
        #                         nn.LeakyReLU(),
        #                         # nn.BatchNorm2d(64),
        #                         Flatten(),
        #                         nn.Linear(in_features=self.conv_output_size, out_features=self.output_size),
        #                         # nn.BatchNorm1d(self.output_size)
        #                         )

    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state
