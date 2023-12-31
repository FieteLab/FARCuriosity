import os
from PIL import Image
import numpy as np
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.models.utils import Flatten
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead
import cv2


class RND(nn.Module):
    """Curiosity model for intrinsically motivated agents: 
    """

    def __init__(
            self,
            image_shape, 
            obs_stats=None,
            prediction_beta=1.0,
            drop_probability=1.0,
            feature_encoding='none',
            gamma=0.99,
            device='cpu',
            minibatches=1,
            target_model=None
            ):
        super(RND, self).__init__()

        self.beta = prediction_beta
        self.decay = 1.0
        self.drop_probability = drop_probability
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        c, h, w = image_shape[0], image_shape[1], image_shape[2]
        if image_shape[0] == 4:
            c = 1
        self.c = c
        self.h = h
        self.w = w
        self.obs_stats = obs_stats
        self.obs_rms = RunningMeanStd(shape=(1, c, h, w)) # (T, B, c, h, w)
        if obs_stats is not None:
            self.obs_rms.mean[0] = obs_stats[0]
            self.obs_rms.var[0] = obs_stats[1]**2
        self.minibatches = minibatches
        self.gamma = gamma
        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(gamma)
        self.step = 0
        self.feature_size = 512
        self.conv_feature_size = 7*7*64

        # Learned predictor model
        self.forward_model = nn.Sequential(
                                            nn.Conv2d(
                                                in_channels=c,
                                                out_channels=32,
                                                kernel_size=8,
                                                stride=4),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=32,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=64,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1),
                                            nn.LeakyReLU(),
                                            Flatten(),
                                            nn.Linear(self.conv_feature_size, self.feature_size),
                                            nn.ReLU(),
                                            nn.Linear(self.feature_size, self.feature_size),
                                            nn.ReLU(),
                                            nn.Linear(self.feature_size, self.feature_size)
                                            )

        for param in self.forward_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()

        # Fixed weight target model
        if target_model is not None:
            self.target_model = target_model
        else:
            self.target_model = nn.Sequential(
                                                nn.Conv2d(
                                                    in_channels=c,
                                                    out_channels=32,
                                                    kernel_size=8,
                                                    stride=4),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(
                                                    in_channels=32,
                                                    out_channels=64,
                                                    kernel_size=4,
                                                    stride=2),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(
                                                    in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1),
                                                nn.LeakyReLU(),
                                                Flatten(),
                                                nn.Linear(self.conv_feature_size, self.feature_size)
                                            )

            for param in self.target_model:
                if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                    nn.init.orthogonal_(param.weight, np.sqrt(2))
                    param.bias.data.zero_()
            for param in self.target_model.parameters():
                param.requires_grad = False


    def init_model(self, init_target=True):

        self.obs_rms = RunningMeanStd(shape=(1, self.c, self.h, self.w)) # (T, B, c, h, w)
        if self.obs_stats is not None:
            self.obs_rms.mean[0] = self.obs_stats[0]
            self.obs_rms.var[0] = self.obs_stats[1]**2

        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(self.gamma)

        for param in self.forward_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()
        
        if init_target:
            for param in self.target_model:
                if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                    nn.init.orthogonal_(param.weight, np.sqrt(2))
                    param.bias.data.zero_()
            for param in self.target_model.parameters():
                param.requires_grad = False




    def forward(self, obs, not_done=None):

        # in case of frame stacking
        if obs.shape[2] == 4:
            obs = obs[:,:,-1,:,:]
            obs = obs.unsqueeze(2)
        obs_cpu = obs.clone().cpu().data.numpy()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        
        if self.device == torch.device('cuda:0') and obs.device.type != 'cpu':
            obs_mean = torch.from_numpy(self.obs_rms.mean).float().cuda()
            obs_var = torch.from_numpy(self.obs_rms.var).float().cuda()
        else:
            obs_mean = torch.from_numpy(self.obs_rms.mean).float()
            obs_var = torch.from_numpy(self.obs_rms.var).float()
        norm_obs = (obs.clone().float() - obs_mean) / (torch.sqrt(obs_var)+1e-10)
        norm_obs = torch.clamp(norm_obs, min=-5, max=5).float()

        # prediction target
        phi = self.target_model(norm_obs.clone().detach().view(T * B, *img_shape)).view(T, B, -1)

        # make prediction
        predicted_phi = self.forward_model(norm_obs.detach().view(T * B, *img_shape)).view(T, B, -1)

        # update statistics
        if not_done is not None:
            valid_obs = np.reshape(obs_cpu[np.where(not_done==1)[:2]], (-1, self.c, self.h, self.w))
            if len(valid_obs) > 0: # avoid all done case.
                self.obs_rms.update(valid_obs)

        return phi, predicted_phi, T

    def compute_bonus(self, next_observation, done, verbose=False, errors=None):
        done = done.cpu().data.numpy()
        not_done = np.abs(done-1)

        if errors is None:
            phi, predicted_phi, T = self.forward(next_observation, not_done=not_done)
            errors = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size

        rewards = errors

        # update running mean
        rewards_cpu = rewards.clone().cpu().data.numpy()
        total_rew_per_env = np.array([self.rew_rff.update(rewards_cpu[i], not_done=not_done[i]) for i in range(T)])
        self.rew_rms.update_from_moments(np.mean(total_rew_per_env), np.var(total_rew_per_env), np.sum(not_done))

        # normalize rewards
        if self.device == torch.device('cuda:0'):
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float().cuda()
            not_done = torch.from_numpy(not_done).float().cuda()
        else:
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float()
            not_done = torch.from_numpy(not_done).float()
        rewards /= max(torch.sqrt(rew_var), 1e-32)


        # apply done mask
        rewards *= not_done

        if verbose:
            return self.decay * self.beta * rewards, errors.detach(), phi.detach()
        return self.decay * self.beta * rewards


    def compute_loss(self, next_observations, valid, indices=None):
        phi, predicted_phi, _ = self.forward(next_observations, not_done=None)
        forward_loss = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size
        mask = torch.rand(forward_loss.shape)
        mask = 1.0 - (mask > self.drop_probability).float().to(self.device)
        net_mask = mask * valid
        loss = torch.sum(forward_loss * net_mask.detach()) / torch.sum(net_mask.detach()).clip(min=1)
        return loss


