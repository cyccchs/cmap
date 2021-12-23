import torch.nn as nn
import network
import agent

class RecurrentAttention(nn.Module):
    def __init__(self, g, k, s, c, h_g, h_l, glimpse_size, std, hidden_size, num_classes, num_agents):
        self.std = std
        for i in range(num_agents):
            self.agents.append(Agent(h_g, h_l, glimpse_size, c, input_size, output_size))

