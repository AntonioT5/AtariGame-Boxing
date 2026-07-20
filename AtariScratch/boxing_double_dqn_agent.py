import os

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

        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x=x/255.0
        x=self.features(x)
        x=x.reshape(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, num_channels=4, num_actions=18, learinging_rate=0.000005, 
                 dicount_factor=0.99, batch_size=32, memory_size=50000, folder_path="LearnedExperience"): #memory_size=50 000 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_channels = num_channels
        self.num_actions = num_actions
        self.dicount_factor=dicount_factor
        self.batch_size=batch_size
        self.memory_size = memory_size
        self.folder_path=folder_path

        os.makedirs(self.folder_path, exist_ok=True) # create folder for knowledge if missing

        self.memory = deque(maxlen=memory_size)

        self.model = BoxingAtariNet(num_channels, num_actions).to(self.device)
        self.target_model = BoxingAtariNet(num_channels,num_actions).to(self.device)

        self.criterion = nn.SmoothL1Loss()
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
        
        state_t = torch.tensor(self.preproecess_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()
        
    def save(self, model_name, episode):
        path = os.path.join(self.folder_path, f'{model_name}_{episode}.pt')
        torch.save(self.model.state_dict(), path)

    def load(self, model_name, episode):
        path = os.path.join(self.folder_path, f'{model_name}_{episode}.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for state, action, reward, next_state, done in batch:
            states.append(self.preproecess_state(state))
            next_states.append(self.preproecess_state(next_state))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        state_tensor = torch.tensor(np.array(states)).to(self.device)
        next_state_tensor = torch.tensor(np.array(next_states)).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q = self.model(state_tensor).gather(1, action_tensor).squeeze(1)

        with torch.no_grad():
            best_actions = torch.argmax(self.model(next_state_tensor), dim=1, keepdim=True)
            next_q = self.target_model(next_state_tensor).gather(1, best_actions).squeeze(1)
            target_q = reward_tensor + self.dicount_factor * next_q * (1 - done_tensor)

        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        return loss.item()


