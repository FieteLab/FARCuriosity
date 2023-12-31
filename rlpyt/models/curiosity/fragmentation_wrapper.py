import copy
import gc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rlpyt.models.curiosity.rnd import RND
from collections import deque


class RunningStats(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, capacity=0):
        self.mean = 0
        self.square_mean = 0
        self.std = 1
        self.count = 0
        self.prev_sample = None
        self.capacity = capacity
        self.prev_mean = 0
        self.memory = deque([],maxlen=capacity*2)

    def update(self, x):
        self.prev_sample = x

        if self.capacity == 0 or len(self.memory) < self.capacity:
            self.mean = (self.mean * self.count + x) / (self.count + 1)
            self.square_mean = (self.square_mean * self.count + x**2) / (self.count + 1)
        else:
            last_item = self.memory[-self.capacity]
            self.mean = self.mean + (x - last_item) / self.capacity
            self.square_mean = self.square_mean + (x**2 - last_item**2) / self.capacity
        self.count += 1
        
        if self.capacity > 0:
            if len(self.memory) == self.capacity:
                self.prev_mean = self.mean
            elif len(self.memory) == 2 * self.capacity - 1:
                self.prev_mean = self.prev_mean - (self.memory[self.capacity] - self.memory[0]) / self.capacity
            elif len(self.memory) == 2 * self.capacity:
                self.prev_mean = self.prev_mean + (self.memory[self.capacity] - self.memory[0]) / self.capacity

        if self.count > 1:
            if self.capacity == 0:
                count = self.count
            else:
                count = min(self.count, self.capacity)
            self.std = (self.square_mean - self.mean ** 2) * count / (count - 1)
            self.std = self.std ** 0.5

        self.memory.append(x)


class FragmentationWrapper(nn.Module):
    def __init__(self, alg, image_shape, output_size, fragmentation_kwargs, curiosity_init_kwargs, obs_stats=None):
        super(FragmentationWrapper, self).__init__()
        self.alg = alg

        device = fragmentation_kwargs['device']
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        self.image_shape = image_shape
        self.output_size = output_size
        self.obs_stats = obs_stats
        self.curiosity_init_kwargs = curiosity_init_kwargs

        self.num_envs = fragmentation_kwargs.get('num_envs', 1)
        self.threshold = fragmentation_kwargs.get('threshold', 2)
        self.recall_threshold = fragmentation_kwargs.get('recall_threshold', 1.0)
        self.use_recall = fragmentation_kwargs.get('use_recall', False)
        self.mem_size = fragmentation_kwargs.get('mem_size', -1)
        self.frag_obs_scale = fragmentation_kwargs.get('frag_obs_scale', 1.0)
        self.cos_th = fragmentation_kwargs.get('cos_th', 1.0)
        self.cos_th_min = fragmentation_kwargs.get('cos_th_min', self.cos_th)
        self.use_feature = fragmentation_kwargs.get('use_feature', False)
        self.frag_criteria = fragmentation_kwargs.get('frag_criteria', 'z')
        self.capacity = fragmentation_kwargs.get('interval', 0)

        curiosity_model = self.get_curiosity_model()
        self.feature_size = getattr(curiosity_model, 'feature_size', None)
        self.curiosity_models = nn.ModuleList([curiosity_model])  # list of curiosity_models

        self.current_idx = [0 for _ in range(self.num_envs)]
        if self.mem_size > 0:
            self.cos_decay = (self.cos_th - self.cos_th_min) / self.mem_size
        else:
            self.cos_decay = (self.cos_th - self.cos_th_min) / 100

        self.frag_info = [] # [[state, prev_idx, next_idx]]
        self.merge_env = True
        if self.merge_env:
            self.dist = [RunningStats(self.capacity)]
        else:
            self.dist = [[RunningStats(self.capacity) for _ in range(self.num_envs)]]

        self.frag_points = [[0 for _ in range(self.num_envs)]]
            
        # for finding the oldest used memory
        self.recent_usage = [0]
        self.step = 0

        self.rewems = [None]

        self.burn_in = [True for _ in range(self.num_envs)]
        self.replace_counter = 0 # for remove cache
        self.clear_cache_interval = 50

        self.std_th = 1e-6
        self.new_models = []

    @property
    def decay(self):
        return self.curiosity_models[0].decay

    def get_curiosity_model(self, curr_idx=None):
        target_model = None
        if self.alg == 'rnd':
            curiosity_model = RND(image_shape=self.image_shape, obs_stats=self.obs_stats, target_model=target_model, **self.curiosity_init_kwargs)
        else:
            raise NotImplementedError
        
        return curiosity_model.to(self.device)


    def compute_bonus(self, input1, input2, input3=None): #next_observation, done):
        num_models = len(self.curiosity_models)
        next_observation = input1
        done = input2

        alloc = self._alloc_each_model()
        reward = []
        features = []
        errors = []
        indices = []
        rewems = np.zeros(self.num_envs)
        for idx, envs in alloc.items():
            curiosity_model = self.curiosity_models[idx]
            next_obs = next_observation[:,envs]
            indices += envs
            if self.alg == 'rnd':
                d = done[:, envs]
                if self.rewems[idx] is not None:
                    curiosity_model.rew_rff.rewems = self.rewems[idx][envs]
                rew, err, feat = curiosity_model.compute_bonus(next_obs, d, verbose=True)
                if self.rewems[idx] is None:
                    rewems = np.zeros(self.num_envs)
                    rewems[envs] = curiosity_model.rew_rff.rewems
                    self.rewems[idx] = rewems
                else:
                    self.rewems[idx][envs] = curiosity_model.rew_rff.rewems
            else:
                raise NotImplementedError

            reward.append(rew)
            errors.append(err)
            features.append(feat)

        indices = torch.tensor(indices, device=next_observation.device)
        reward = torch.cat(reward, -1)
        errors = torch.cat(errors, -1)
        features = torch.cat(features, 1)
        features = features.index_select(1, indices)
        reward = reward.index_select(1, indices)
        errors = errors.index_select(1, indices)

        if self.frag_obs_scale != 1:
            T, E, C, H, W = next_observation.shape
            next_obs = next_observation.view(T*E, C, H, W).float()
            H = int(H / self.frag_obs_scale)
            W = int(W / self.frag_obs_scale)
            next_obs = F.interpolate(next_obs, (H, W), mode='bilinear')
            next_obs = next_obs.view(T, E, C, H, W)
        else:
            next_obs = next_observation
    
        if self.use_feature:
            target = features
        else:
            next_obs = next_obs.detach().float().mean(2, keepdim=True)
            target = next_obs
        
        if len(self.frag_info) == 0:
            self.frag_info.append([target[0,0].view(-1).float(), 0, 0])

        if len(self.curiosity_models) >= 1 and self.use_recall:
            recall_info, max_cos_sim, max_idx = self.check_recall(target)
        else:
            recall_info = None
            max_cos_sim = None
            max_idx = None
        errors = self.fragmentation(errors, target, done, recall_info, max_cos_sim, max_idx, next_obs)
        self.cos_th = max(self.cos_th_min, self.cos_th - (len(self.curiosity_models) -1) * self.cos_decay)

        new_num_models = len(self.curiosity_models)
        self.new_models += list(range(num_models, new_num_models))

        return reward

    def get_new_model_idx(self):
        out = copy.deepcopy(self.new_models)
        self.new_models = []
        return out

    def compute_loss(self, input1, input2, input3=None, input4=None, input5=None):
        if self.alg == 'rnd':
            next_observations = input1
            valid = input2
            indices = input3

        alloc = self._alloc_each_model(indices)
        loss = 0
        for idx, envs in alloc.items():
            curiosity_model = self.curiosity_models[idx]
            next_obs = next_observations[:,envs]
            val = valid[:,envs]
            loss += curiosity_model.compute_loss(next_obs, val)
        return loss

    def _alloc_each_model(self, indices=None): 
        """
            Check whether currently how many 
        """
        alloc = {}
        for i, idx in enumerate(self.current_idx):
            if indices is not None:
                if i not in indices:
                    continue
                t = int(np.where(indices==i)[0])
            else:
                t = i
            if idx not in alloc:
                alloc[idx] = [t]
            else:
                alloc[idx].append(t)
        return alloc


    def recall(self, idx, n, i, recall_info, next_obs, other_inputs, errors, is_local=False):
        if idx != int(recall_info[1]) and int(recall_info[1]) > -1:
            new_idx = int(recall_info[1])
        else: # going back to the previous memory
            new_idx = idx
        
        if new_idx != idx:
            idx = new_idx
            if self.merge_env:
                self.frag_points[idx][n] = self.dist[idx].count
            else:
                self.frag_points[idx][n] = self.dist[idx][n].count
            self.recent_usage[idx] = self.step + i
            self.current_idx[n] = idx
        return idx, errors


    def _get_stat(self, idx, n):
        if self.capacity == 0:
            if self.merge_env:
                return self.dist[idx].mean, self.dist[idx].std
            else:
                return self.dist[idx][n].mean, self.dist[idx][n].std
        else:
            if self.merge_env:
                return self.dist[idx].mean, self.dist[idx].prev_mean
            else:
                return self.dist[idx][n].mean, self.dist[idx][n].prev_mean

    def fragmentation(self, errors, next_observations, other_inputs, recall_info, max_cos_sim=None, max_idx=None, original_obs=None):
        local_frag_info = []
        min_length = max(self.capacity*2+1, 25)
        if original_obs is None:
            original_obs = next_observations

        for n in range(self.num_envs):
            if self.burn_in[n]:
                self.burn_in[n] = False
                continue
            idx = self.current_idx[n]
            if len(local_frag_info) > 0:
                local_frag_info = [e for e in local_frag_info if len(e) == 3]
                local_recall_info, local_max_cos_sim, local_max_idx = self.check_recall(next_observations[:,n:n+1], local_frag_info)
            else:
                local_recall_info, local_max_cos_sim = None, None
            recalled = False
            
            for i in range(errors.shape[0]): # horizon
                # Same Observation.
                if self.merge_env:
                    if errors[i,n] == self.dist[idx].prev_sample:
                        continue
                else:
                    if errors[i,n] == self.dist[idx][n].prev_sample:
                        continue

                error = errors[i,n]
                mean, std = self._get_stat(idx, n)
                if self.frag_criteria == 'z':
                    is_frag = error - mean >= self.threshold * std # fragmentation
                elif self.frag_criteria == 'ratio': # ratio
                    is_frag = error >= self.threshold * mean # fragmentation
                else:
                    raise NotImplementedError
                # previous recall info
                if not recalled:
                    if recall_info is not None and recall_info[i, n, 1] > -1:
                        new_idx, errors = self.recall(idx, n, i, recall_info[i,n], next_observations, other_inputs, errors)
                        if new_idx != idx:
                            idx = new_idx
                            recalled = True
                    # recently created model in a same minibatch
                    elif local_recall_info is not None and local_recall_info[i, 0, 1] > -1:
                        new_idx, errors = self.recall(idx, n, i, local_recall_info[i,0], next_observations, other_inputs, errors, is_local=True)
                        if new_idx != idx:
                            idx = new_idx
                            break
                
                if recalled: # current error does not come from the recalled model.
                    break

                self.recent_usage[idx] = self.step + i # update recent usage 
                
                if self.merge_env and self.dist[idx].count < min_length:
                    self.dist[idx].update(error.item())
                    continue
                elif not self.merge_env and self.dist[idx][n].count < min_length:
                    self.dist[idx][n].update(error.item())
                    continue

                if std <= self.std_th: # might be aggregated from single location
                    continue

                if is_frag:
                    cos_sim = 0.0
                    if max_cos_sim is not None:
                        if max_cos_sim[i, n] >= self.cos_th or next_observations[i,n].sum() == 0:
                            continue
                        cos_sim = max_cos_sim[i,n].item()

                        if local_max_cos_sim is not None:
                            if local_max_cos_sim[i, 0] >= self.cos_th:
                                continue
                            cos_sim = max(cos_sim, local_max_cos_sim[i, 0].item())

                    new_idx = self.create_new_model(idx)
                    self.frag_info.append([next_observations[i,n].view(-1).float().clone(), self.current_idx[n], new_idx]) # use gpu
                    self.current_idx[n] = new_idx
                    self.burn_in[n] = True
                    local_frag_info.append(self.frag_info[-1])
                    idx = new_idx
                    break

                if self.merge_env:
                    self.dist[idx].update(error.item())
                else:
                    self.dist[idx][n].update(error.item())


        self.step += int(errors.shape[0])
        return errors


    def create_new_model(self, curr_idx):
        """
            Create a new local curiosity module
        """
        if self.mem_size == -1 or self.mem_size > len(self.curiosity_models):
            curiosity_model = self.get_curiosity_model(curr_idx)
            self.curiosity_models.append(curiosity_model)
            if self.merge_env:
                self.dist.append(RunningStats(self.capacity))
            else:
                self.dist.append([RunningStats(self.capacity) for _ in range(self.num_envs)])
            
            self.frag_points.append([0 for _ in range(self.num_envs)])
            self.recent_usage.append(self.step + 1)
            if self.alg == 'rnd':
                self.rewems.append(None)
            return len(self.curiosity_models) - 1
        # replace existing memory
        oldest = np.argmin(self.recent_usage)

        # init the network
        self.curiosity_models[oldest].init_model() 

        temp = self.dist[oldest]
        self.dist[oldest] = None
        del temp

        if self.merge_env:
            self.dist[oldest] = RunningStats(self.capacity)
        else:
            self.dist[oldest] = [RunningStats(self.capacity) for _ in range(self.num_envs)]

        self.frag_points[oldest] = [0 for _ in range(self.num_envs)]
        self.recent_usage[oldest] = self.step + 1
        if self.alg == 'rnd':
            self.rewems[oldest] = None
        
        # modify the fragmentation.
        i = 0

        while i < len(self.frag_info): # the size of the info can be dynamically changed.
            if self.frag_info[i][2] == oldest:
                del self.frag_info[i][0]
                del self.frag_info[i]
                continue
            elif self.frag_info[i][1] == oldest:
                self.frag_info[i][1] = -1
            i += 1

        self.replace_counter += 1 

        if self.replace_counter % self.clear_cache_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return oldest


    def check_recall(self, next_observations, frag_info=None):
        T, N = next_observations.shape[:2] # time horizon, n_env
        states = next_observations.view(T*N, -1).float()
        recall_info = {}

        recall_info_prev = -torch.ones((T*N,1))
        recall_info_next = -torch.ones((T*N,1))
        if frag_info is None:
            frag_info = self.frag_info
        
        frag_states = torch.stack([e[0] for e in frag_info]).T.unsqueeze(0)
        
        div = 1
        with torch.no_grad():
            cos_sim = [F.cosine_similarity(states.unsqueeze(-1), frag_states[:,:,div*i:div*(i+1)]) for i in range(len(frag_info)//div + 1)]
        cos_sim = torch.cat(cos_sim, -1)
        max_cos_sim, max_idx = cos_sim.max(dim=1)
        ret_max_idx = max_idx.clone()

        non_recall = max_cos_sim < self.recall_threshold
        max_idx[non_recall] = -1
        max_idx = max_idx.view(-1)
        unique_idx = max_idx.unique()
        for idx in unique_idx.view(-1):
            if idx < 0:
                continue
            recall_info_prev[max_idx == idx] = frag_info[idx][1]
            recall_info_next[max_idx == idx] = frag_info[idx][2]
        
        recall_info = torch.cat((recall_info_prev, recall_info_next), -1)
        recall_info = recall_info.view(T, N, -1)
        return recall_info, max_cos_sim.view(T,N), ret_max_idx.view(T,N)
