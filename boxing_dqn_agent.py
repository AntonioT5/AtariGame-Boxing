import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class BoxingAtariNet(nn.Module):
    def __init__(self, channels, actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, actions)
        )

    def forward(self, x):
        x=x/255.0
        x=self.features(x)
        x=x.reshape(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, num_channels=4, num_actions=18, learinging_rate=0.0001, 
                 dicount_factor=0.99, batch_size=32, memory_size=50000):
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.dicount_factor=dicount_factor
        self.batch_size=batch_size
        self.memory_size = memory_size

        self.memory = deque(maxlen=memory_size)

        self.model = BoxingAtariNet(num_channels, num_actions)
        self.target_model = BoxingAtariNet(num_channels,num_actions)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learinging_rate)
        self.update_target_model()

    def preproecess_state(self, obs):
        # obs comes with (84, 84, 4) after our preprocessing class and now for conv2d we need (4, 84, 84)

        obs = np.array(obs, dtype=np.float32)
        obs = np.transpose(obs, (2,0,1))
        return obs
    
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        
        state_t = torch.tensor(self.preproecess_state(state)).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()
        
    def save(self, model_name, episode):
        torch.save(self.model.state_dict(), f'{model_name}_{episode}.pt')

    def load(self, model_name, episode):
        self.model.load_state_dict(torch.load(f'{model_name}_{episode}.pt'))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            state_t = torch.tensor(self.preproecess_state(state)).unsqueeze(0)
            next_state_t = torch.tensor(self.preproecess_state(next_state)).unsqueeze(0)

            target = self.model(state_t).detach().clone().squeeze()

            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    max_future_q = torch.max(self.target_model(next_state_t))
                target[action] = reward + self.dicount_factor * max_future_q

            states.append(state_t)
            targets.append(target)

        state_tensor = torch.cat(states, dim=0)
        target_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(state_tensor)
        loss = self.criterion(outputs, target_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()


