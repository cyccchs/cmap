import torch.nn as nn
import torch
import networks
import agent
from torch.autograd import Variable

class RecurrentAttention(nn.Module):
    def __init__(self, batch_size, h_g, h_l, glimpse_size, c, hidden_size, glimpse_num):
        super().__init__()
        
        self.init_l = torch.FloatTensor(batch_size, 2).uniform_(-1.0,1.0)
        self.init_h = torch.zeros((4, hidden_size), dtype=torch.float32)
        self.hidden_size = hidden_size
        self.retina = networks.GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.selfatt = networks.SelfAttention()
        self.softatt = networks.SoftAttention()
        self.core = networks.CoreNetwork(batch_size, hidden_size, glimpse_num) 
    def forward(self, img, existence):
        g_t = self.retina(img, self.init_l)
        s_t = self.selfatt(g_t)
        alpha, z_t = self.softatt(g_t, self.init_h)

        lstm = nn.LSTMCell(4, self.hidden_size)
        lstm_out, state = lstm(z_t)
        lstm_out, state = lstm(z_t, (lstm_out,state))
        
        return g_t


