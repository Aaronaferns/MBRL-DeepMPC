import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, num_states):
        super(ValueNetwork, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(num_states, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
        # Initialize weights for the layers. Ignoring the bias of the fc layers
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu') 
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu') 
        nn.init.uniform_(self.out.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(self.out.bias, a=-3e-3, b=3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.out(x)  # Output layer with linear activation
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepValueNetwork(nn.Module):
    def __init__(self, num_states):
        super(DeepValueNetwork, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        
    
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu') 
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.uniform_(self.out.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(self.out.bias, a=-3e-3, b=3e-3)

    def forward(self, state):
        # Forward pass with ReLU activations and dropout
        x = F.relu(self.fc1(state))
      
        x = F.relu(self.fc2(x))
  
        x = F.relu(self.fc3(x))
  
        x = F.relu(self.fc4(x))
        output = self.out(x)  # Output layer with linear activation
        return output
