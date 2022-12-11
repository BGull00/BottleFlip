<<<<<<< HEAD


# Import Required Packages
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from replay_memory import ReplayBuffer

# Use CPU or GPU computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

     # Initialize agent with parameters for DQN
    def __init__(self, state_size, action_size, dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99, learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):
        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)

        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)


    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))






=======


# Import Required Packages
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from replay_memory import ReplayBuffer

# Use CPU or GPU computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

     # Initialize agent with parameters for DQN
    def __init__(self, state_size, action_size, dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99, learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):
        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)

        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)


    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))






>>>>>>> e970ba12a2b44054a6155842085cc318c27a79f9
