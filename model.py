import torch.nn as nn
import torch
import networks
import agent

class RecurrentAttention(nn.Module):
    def __init__(self, batch_size, h_g, h_l, glimpse_size, c, lstm_size):
        super().__init__()
        
        self.init_l = torch.FloatTensor(batch_size, 2).uniform_(-1.0,1.0)
        self.init_h = torch.zeros((batch_size, lstm_size), dtype=torch.float32)

        self.retina = networks.GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.selfatt = networks.SelfAttention()
        self.softatt = networks.SoftAttention()
    
    def forward(self, img):
        g_t = self.retina(img, self.init_l)
        s_t = self.selfatt(g_t)
        alpha, z_t = self.softatt(g_t, init_h)
        return g_t


