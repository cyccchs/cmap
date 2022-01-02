import torch.nn as nn
import torch
import networks
import agent

class RecurrentAttention(nn.Module):
    def __init__(self, h_g, h_l, glimpse_size, c):
        super().__init__()
        self.init_l = torch.FloatTensor(4,2).uniform_(-1.0,1.0)
        self.glimpsenn = networks.GlimpseNetwork(h_g, h_l, glimpse_size, c)
    def forward(self, img):
        return(self.glimpsenn(img, self.init_l))
        
