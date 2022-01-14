import torch.nn as nn
import torch
import networks
from agent import Agent

class RecurrentAttention(nn.Module):
    def __init__(self, batch_size, h_g, h_l, glimpse_size, c, lstm_size, hidden_size, loc_dim, std):
        super().__init__()
        self.agents = []
        self.agent_num = 8
        
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.retina = networks.GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.location = networks.LocationNetwork(256, 2, std=0.22)
        self.baseline = networks.BaselineNetwork(256, 1)
        self.selfatt = networks.SelfAttention()
        self.softatt = networks.SoftAttention()
        self.lstm = networks.CoreNetwork(lstm_size)
        self.classifier = networks.ActionNetwork(256, 2)
        for i in range(self.agent_num):
            self.agents.append(Agent(h_g, h_l, glimpse_size, c, hidden_size, loc_dim, std))
    
    def forward(self, img, h_t, l_t, last=False):
        g_list = []
        for i in range(self.agent_num):
            g_list.append(self.agents[i].glimpse_feature(img, l_t))
        s_t = self.selfatt(g_list)
        alpha, z_t = self.softatt(g_list, h_t)
        h_t = self.lstm(z_t)
        log_pi, l_t = self.location(s_t)
        b_list = self.baseline(s_t)
        
        #print('g_t', g_t.shape)
        #print('s_t', s_t.shape)
        #print('alpha', alpha.shape)
        #print('h_t', h_t.shape)
        #print('l_t', l_t.shape)
        #print('b_t', len(b_list))
        #print('log_pi', log_pi.shape)
        
        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_list, log_pi, log_probas
        
        return h_t, l_t, b_list, log_pi

