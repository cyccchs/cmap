import torch.nn as nn
import torch
import networks
import agent
from torch.autograd import Variable

class RecurrentAttention(nn.Module):
    def __init__(self, batch_size, h_g, h_l, glimpse_size, c, lstm_size):
        super().__init__()
        
        self.init_l = torch.FloatTensor(batch_size, 2).uniform_(-1.0,1.0)
        self.init_h = torch.zeros((4, lstm_size), dtype=torch.float32)
        self.lstm_size = lstm_size

        self.retina = networks.GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.selfatt = networks.SelfAttention()
        self.softatt = networks.SoftAttention()
        self.location = networks.LocationNetwork(256, 2, std=0.22)
        self.lstm = networks.CoreNetwork(lstm_size)
    
    def forward(self, img, existence):
        g_t = self.retina(img, self.init_l)
        s_t = self.selfatt(g_t)
        alpha, z_t = self.softatt(g_t, self.init_h)
        h_t = self.lstm(z_t)
        log_pi, l_t = self.location(h_t)
        
        return g_t


