import os

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models


class PretrainedBoxingNet(nn.Module):
    def __init__(self, channels, actions):
        super().__init__()
       
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(
            channels, 
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride= old_conv1.stride,
            padding=old_conv1.padding,
            bias=(old_conv1.bias is not None)
        )

        with torch.no_grad():
            avg_weight = old_conv1.weight.mean(dim=1, keepdim=True)

            new_conv1.weight[:] = avg_weight.repeat(1, channels, 1, 1)

        resnet.conv1=new_conv1

        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, actions)

        for name, param in resnet.named_parameters():
            if name.startswith("bn1") or name.startswith("layer1") or name.startswith("layer2") or name.startswith("layer3"):
                param.requires_grad = False

        self.resnet = resnet

    def forward(self, x):
        x=x/255.0
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-5)
        return self.resnet(x)
    
    def train(self, mode=True):
        super().train(mode)
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.BatchNorm2d) and (name.startswith("bn1") or name.startswith("layer1") or name.startswith("layer2")) or name.startswith("layer3"):
                module.eval()


class PretrainedDQNAgent:
    def __init__(self, num_channels=4, num_actions=18, learinging_rate=0.000005,
                 dicount_factor=0.99, batch_size=32, memory_size=50000, folder_path="LearnedExperiencePretrained"): #memory_size=200 000, learing_rate=0.00001 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_channels = num_channels
        self.num_actions = num_actions
        self.dicount_factor=dicount_factor
        self.batch_size=batch_size
        self.memory_size = memory_size
        self.folder_path=folder_path

        os.makedirs(self.folder_path, exist_ok=True) # create folder for knowledge if missing

        self.memory = deque(maxlen=memory_size)

        self.model = PretrainedBoxingNet(num_channels, num_actions).to(self.device)
        self.target_model = PretrainedBoxingNet(num_channels,num_actions).to(self.device)
        self.target_model.eval()

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learinging_rate)
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
        
        self.model.eval()
        
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
        
        self.model.train()
        
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


