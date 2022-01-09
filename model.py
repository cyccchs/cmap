import torch.nn as nn
import torch
import networks
import agent
from torch.autograd import Variable

class RecurrentAttention(nn.Module):
    def __init__(self, batch_size, h_g, h_l, glimpse_size, c, lstm_size):
        super().__init__()
        
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.retina = networks.GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.selfatt = networks.SelfAttention()
        self.softatt = networks.SoftAttention()
        self.location = networks.LocationNetwork(256, 2, std=0.22)
        self.lstm = networks.CoreNetwork(lstm_size)
        self.baseline = networks.BaselineNetwork(256, 1)
    
    def forward(self, img, existence):
        l_t, h_t = self.reset()

        for i in range(4-1):
            g_t = self.retina(img, l_t)
            b_t = self.selfatt(g_t)
            alpha, z_t = self.softatt(g_t, h_t)
            h_t = self.lstm(z_t)
            log_pi, l_t = self.location(h_t)
            b_t = self.baseline(b_t)

        return g_t

    def reset(self):
        init_l = torch.FloatTensor(self.batch_size, 2).uniform_(-1.0,1.0)
        init_h = torch.zeros((4, self.lstm_size), dtype=torch.float32)

        return init_l, init_h

