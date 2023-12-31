import logging
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.models.curiosity.encoders import UniverseHead, BurdaHead, MazeHead
from rlpyt.models.curiosity.forward_models import *

INF = float('inf')

class EpisodicCuriosity(nn.Module):
    def __init__(
          self, 
          image_shape, 
          action_size,
          memory_size,
          num_envs,
          n_nearest_neighbors=10,
          cluster_distance=0.008,
          eps=1e-3,
          c=1e-2,
          max_similarity=8.0,
          feature_encoding='idf_burda', 
          batch_norm=False,
          prediction_beta=1.0,
          obs_stats=None
          ):
      """Episodic curiosity model (see https://arxiv.org/pdf/2002.06038.pdf)

      Args:
          image_shape ([tuple]): input image shape (C, H, W)
          action_size ([int]): number of disrete actions in the action space (NOTE: not support continuous actions for now)
          memory_size ([int]): episodic memory size (number of memory slots)
          num_envs (int): number of parallel environments (we need this for create the episodic memory buffer)
          n_nearest_neighbors (int, optional): number of nearest neighbors. Defaults to 10.
          cluster_distance (float, optional): cluster distance (see the paper). Defaults to 0.008.
          eps ([float], optional): a small constant (see the paper). Defaults to 1e-3.
          c ([float], optional): a small constant (see the paper). Defaults to 1e-2.
          max_similarity (float, optional): a cutoff threshold for similarity (zero if exceed). Defaults to 8.0.
          feature_encoding (str, optional): type of encoder. Defaults to 'idf_burda'.
          batch_norm (bool, optional): flag for batch normalization in the encoder. Defaults to False.
          prediction_beta (float, optional): weight coefficient of curiosity reward. Defaults to 1.0.
          obs_stats (RunningMeanStd, optional): for normalizing the observation (usually used in MuJoCo). Defaults to None.
      """
      super(EpisodicCuriosity, self).__init__()
      self.image_shape = image_shape
      self.num_envs = num_envs
      self.memory_size = memory_size
      self.num_envs = num_envs
      self.n_nearest_neighbors = n_nearest_neighbors
      self.cluster_distance = cluster_distance
      self.eps = eps
      self.c = c
      self.max_similarity = max_similarity
      self.prediction_beta = prediction_beta
      self.feature_encoding = feature_encoding
      self.obs_stats = obs_stats
      if self.obs_stats is not None:
          self.obs_mean, self.obs_std = self.obs_stats
      
      self.inverse_loss_wt = 1.0

      if self.feature_encoding != 'none':
          if self.feature_encoding == 'idf':
              self.feature_size = 288
              self.encoder = UniverseHead(image_shape=image_shape, batch_norm=batch_norm)
          elif self.feature_encoding == 'idf_burda':
              self.feature_size = 512
              self.encoder = BurdaHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)
          elif self.feature_encoding == 'idf_maze':
              self.feature_size = 256
              self.encoder = MazeHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)

      self.inverse_model = nn.Sequential(
          nn.Linear(self.feature_size * 2, self.feature_size),
          nn.ReLU(),
          nn.Linear(self.feature_size, action_size)
          )

      self.reset_all()
    
    def reset_all(self):
      """Reset memory and its related variables for all parallel environments.
      """
      # TODO: make sure model.to(device) work on this
      # Memory is a circular buffer
      self.memory = torch.zeros((self.num_envs, self.memory_size, self.feature_size))    
      # How many elements are in the memory
      self.n_elements_in_memory = np.zeros(self.num_envs, dtype=np.int32)
      # The position of the cursor
      self.cursor = np.zeros(self.num_envs, dtype=np.int32)
      # Running average of euclidean distance of k-nearest neighbors (d^2_m in the paper)
      self.knn_distance_running_mean = torch.zeros((self.num_envs, self.n_nearest_neighbors,))
      # Number of kNN queries for calculating running mean
      self.n_knn_queries = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, env_idx):
      self.memory[env_idx] = torch.zeros((self.memory_size, self.feature_size))  
      self.n_elements_in_memory[env_idx] = 0
      self.cursor[env_idx] = 0
      self.knn_distance_running_mean[env_idx] = torch.zeros((self.n_nearest_neighbors,))
      self.n_knn_queries[env_idx] = 0   

    def _encode(self, observation):
      """Encode a batch of observation by self.enoder

      Args:
          observation ([torch.Tensor]): Raw observations in torch.Tensor form. 
                    The dim can be (T, B,) + image_shape or (B,) + image_shape

      Returns:
          [torch.Tensor]: Encoded observation of which size is self.feature_size
      """
      img = observation
      if self.obs_stats is not None:
        img = (observation - self.obs_mean) / self.obs_std         

      img = img.type(torch.float)
    
      # Infer (presence of) leading dimensions: [T,B], [B], or [].
      # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
      lead_dim, T, B, img_shape = infer_leading_dims(observation, 3) 

      phi = img

      if self.feature_encoding != 'none':
          phi = self.encoder(img.view(T * B, *img_shape))      
          phi = phi.view(T, B, -1)
      
      return phi

    def append(self, observations):
      """Append and encode the observation in each parallel environment (i.e., controllable state f(x) in the paper) to the memory.
        The shape of `observations` must be in (T, B,) + obs_size.

      Args:
          observation ([torch.Tensor]): A list of raw observations (dim is assumed to be (T, B,) or (B,) + image_shape)
      """
      assert observations.shape[0] == self.num_envs # T must be equal to self.num_envs
      assert len(observations.shape) == (2 + len(self.image_shape)) # (T, B,) + image_shape

      encoded_states = self._encode(observations)

      T, B = encoded_states.shape[:2]
      assert T == self.num_envs

      for env_idx in range(T):
        for batch_idx in range(B):
          self.memory[env_idx, self.cursor[env_idx]] = encoded_states[env_idx, batch_idx]
          self.cursor[env_idx] = (self.cursor[env_idx] + 1) % self.memory_size
          self.n_elements_in_memory[env_idx] = min(self.n_elements_in_memory[env_idx] + 1, self.memory_size)
          logging.debug('env_idx={}; memory_size={}; n_elements_in_memory={}; cursor={};'.format(env_idx, self.memory_size, self.n_elements_in_memory[env_idx], self.cursor[env_idx]))

    def compute_episodic_curiosity_reward(self, observations):
      """Compute the episodic curiosity reward r^{episodic}_t by k-nearest negihbor lookup in the self.memory

      Args:
          observation ([torch.Tensor]): A batch of observations in each parallel environment. 
                    The shape is expected to be (T, B,) + image_shape.                 

      Returns:
          [np.ndarray]: Episodic curiosity reward r^{episodic}_t for each parallel environment.
                  The shape is (T, B,)
      """
         
      encoded_states = self._encode(observations)

      T, B = encoded_states.shape[:2]
      assert T == self.num_envs

      # TODO: what's a better initialization for each parallel environment when memory of that environment is empty?
      #       for now, I initialize them as one since we 1.0 means no changes in multiplication (i.e., multiply with lifelong curiosity)
      similarities = np.ones((T, B))

      for env_idx in range(T):
        available_memory_size = min(self.memory_size, self.n_elements_in_memory[env_idx])
        if available_memory_size == 0:
          continue
        for batch_idx in range(B):         
          # Retrieve N_k and d(f(x_t), N_k[i])
          euclidean_distances = torch.sqrt(torch.sum((self.memory[env_idx, :available_memory_size, :] - encoded_states[env_idx, batch_idx, ...])**2, dim=-1))
          knn_elements_indices = torch.argsort(euclidean_distances, descending=True, dim=-1)[:self.n_nearest_neighbors]
          
          # Compute d_k[i] = d^2(f(x_t), N_k[i])
          knn_distances = euclidean_distances[knn_elements_indices]**2 # NOTE: they use squared euclidean distance
          
          # Update the running average of distances (d^2_m)
          self.n_knn_queries[env_idx] += 1
          self.knn_distance_running_mean[env_idx] += (1.0 / self.n_knn_queries[env_idx]) * (knn_distances - self.knn_distance_running_mean[env_idx])

          # Normalize the distance d_n = d_k / d^2_m
          normalized_knn_distances = knn_distances / self.knn_distance_running_mean[env_idx]

          # Cluster distance
          normalized_knn_distances = torch.clamp(normalized_knn_distances - self.cluster_distance, min=0.0, max=INF)

          # Compute kernel values K_v
          kernel_values = self.eps / (normalized_knn_distances + self.eps)

          # Compute similarity scores between f(x_t) and N_k
          similarity = (torch.sqrt(torch.sum(kernel_values)) + self.c).detach().cpu().numpy()

          if similarity > self.max_similarity:
            similarities[env_idx, batch_idx] = 0
          else:
            similarities[env_idx, batch_idx] = 1.0 / similarity

      return similarities 

    def forward(self, obs1, obs2):
        img1 = obs1
        img2 = obs2

        if self.obs_stats is not None:
            img1 = (obs1 - self.obs_mean) / self.obs_std
            img2 = (obs2 - self.obs_mean) / self.obs_std

        img1 = img1.type(torch.float)
        img2 = img2.type(torch.float) # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs1, 3) 

        phi1 = img1
        phi2 = img2
        if self.feature_encoding != 'none':
            phi1 = self.encoder(img1.view(T * B, *img_shape))
            phi2 = self.encoder(img2.view(T * B, *img_shape))
            phi1 = phi1.view(T, B, -1)
            phi2 = phi2.view(T, B, -1)

        predicted_action = self.inverse_model(torch.cat([phi1, phi2], 2))
        
        return phi1, predicted_action

    def compute_loss(self, observations, next_observations, actions, valid):
      # dimension add for when you have only one environment
      if actions.dim() == 2: 
        actions = actions.unsqueeze(1)
      phi1, predicted_action = self.forward(observations, next_observations)
      actions = torch.max(actions.view(-1, *actions.shape[2:]), 1)[1] # convert action to (T * B, action_size)     
      inverse_loss = nn.functional.cross_entropy(predicted_action.view(-1, *predicted_action.shape[2:]), actions.detach(), reduction='none').view(phi1.shape[0], phi1.shape[1])    
      inverse_loss = valid_mean(inverse_loss, valid.detach() if valid is not None else None)     
      
      return self.inverse_loss_wt * inverse_loss

if __name__ == '__main__':
  import numpy as np
  import torch
  
  #logger = logging.getLogger()
  #logger.setLevel(logging.DEBUG)
  
  T = 8
  B = 10
  obs_size = (3, 84, 84)
  action_size = 5
  memory_size = 64

  batch_obses = torch.from_numpy(np.random.random((T, B,) + obs_size)).float()
  batch_acts = torch.stack([torch.from_numpy(np.stack([np.eye(action_size)[np.random.randint(low=0, high=action_size)] for i in range(B)])).float() for t in range(T)])
  batch_next_obses = torch.from_numpy(np.random.random((T, B,) + obs_size)).float()
  model = EpisodicCuriosity(image_shape=obs_size,
                action_size=action_size, memory_size=memory_size, num_envs=T)

  # Test training
  optimizer = torch.optim.Adam(model.parameters())
  loss = model.compute_loss(batch_obses, batch_next_obses, batch_acts, valid=None)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  assert isinstance(loss.item(), float)
  print('"training" test passed')

  # Test append
  for t in range(memory_size + 5):
    obs = torch.from_numpy(np.random.random((T, 1,) + obs_size)).float()
    model.append(obs)
    for i in range(T):
      assert model.n_elements_in_memory[i] == min((t + 1), memory_size)
      assert model.cursor[i] == (t + 1) % memory_size
      pass
  print('"append" test passed')

  # Test reward computation
  for t in range(10):
    obs = torch.from_numpy(np.random.random((T, 4,) + obs_size)).float()
    rew = model.compute_episodic_curiosity_reward(obs)
    assert rew.shape == (T, 4)
  print('"compute_episodic_curisoity_reward" test passed')

  # Test reset by index
  for i in range(T):
    model.reset(i)

  # Test reward computation again after reset
  for t in range(T):
    obs = torch.from_numpy(np.random.random((T, 4,) + obs_size)).float()
    rew = model.compute_episodic_curiosity_reward(obs)
    assert rew.shape == (T, 4)
    assert np.all(rew[t] == 1.0)
  print('"compute_episodic_curisoity_reward (empty memory)" test passed')
  
  # Test append again after reset
  for t in range(memory_size + 5):
    obs = torch.from_numpy(np.random.random((T, 1,) + obs_size)).float()
    model.append(obs)
    for i in range(T):
      assert model.n_elements_in_memory[i] == min((t + 1), memory_size)
      assert model.cursor[i] == (t + 1) % memory_size
      pass
  print('"append (after reset)" test passed')

  # Test reward computation again after reset
  for t in range(10):
    obs = torch.from_numpy(np.random.random((T, 4,) + obs_size)).float()
    rew = model.compute_episodic_curiosity_reward(obs)
    assert rew.shape == (T, 4)
  print('"compute_episodic_curisoity_reward (after reset)" test passed')



  