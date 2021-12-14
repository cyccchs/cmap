import torch.nn as nn
import network

class RecurrentAttention(nn.Module):
    def __init__(self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes):
        self.std = std
        self.sensor = network.GlimpseNetwork(h_g, h_l, glimpse_size, c)
