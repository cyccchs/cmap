import torch.nn as nn
import torch
import networks
from agent import Agent

debug = False

class MultiAgentRecurrentAttention(nn.Module):
    def __init__(self, batch_size, agent_num, h_g, h_l, glimpse_size, c, lstm_size, hidden_size, loc_dim, std):
        super().__init__()
        self.agents = []
        self.agent_num = agent_num
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.selfatt = networks.SelfAttention()
        self.softatt = networks.SoftAttention()
        self.lstm = networks.CoreNetwork(batch_size, lstm_size)
        #self.lstm.register_backward_hook(self.fun)
        self.classifier = networks.ActionNetwork(hidden_size, 2)
        for i in range(self.agent_num):
            self.agents.append(Agent(h_g, h_l, glimpse_size, c, hidden_size, loc_dim, std))
    
    def forward(self, img, h_t, l_t, last=False):
        g_list, b_list, l_list, log_pi_list = [], [], [], []
        
        for i in range(self.agent_num):
            g_list.append(self.agents[i].glimpse_feature(img, l_t[i]))
        #g_list: len = agent_num, [b, hidden_size] in each element size
        
        s_t = self.selfatt(g_list)
        s_t = torch.unbind(s_t, dim=1) # s_t: agent_num*(batch_size, hidden_size)
        #alpha, z_t = self.softatt(g_list, h_t)
        #h_t = self.lstm(z_t)
        tempG = torch.cat(g_list, dim=0)
        
        for i in range(self.agent_num):
            log_pi, l_t = self.agents[i].location(s_t[i])
            b = self.agents[i].baseline(s_t[i])
            b_list.append(b)
            l_list.append(l_t)
            log_pi_list.append(log_pi)
        
        b_t = torch.stack(b_list, dim=1) #[agent_num, batch_size]
        log_pi_t = torch.stack(log_pi_list, dim=1)
        
        if debug:
            print('s_t', s_t.shape)
            print('z_t', z_t.shape)
            print('alpha', alpha.shape)
            print('h_t', h_t.shape)
            print('l_t', l_t.shape)
            print('b_t', len(b_list))
            print('log_pi', log_pi.shape)
        
        if last:
            #log_probas = self.classifier(h_t)
            log_probas = self.classifier(tempG)
            log_probas = torch.mean(log_probas, dim=0)
            log_probas = log_probas.repeat(3,1)
            return h_t, l_list, b_t, log_pi_t, log_probas#, alpha
        
        return h_t, l_list, b_t, log_pi_t

