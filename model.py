import torch.nn as nn
import torch
import networks
from agent import Agent

debug = False

class MultiAgentRecurrentAttention(nn.Module):
    def __init__(self, batch_size, agent_num, h_g, h_l, glimpse_size, glimpse_num, c, lstm_size, hidden_size, loc_dim, std, device):
        super().__init__()
        self.agents = []
        self.agent_num = agent_num
        self.glimpse_num = glimpse_num
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.device = device
        self.selfatt = networks.SelfAttention(hidden_size)
        self.softatt = networks.SoftAttention(hidden_size, device)
        #self.lstm = networks.CoreNetwork(batch_size, lstm_size, device)
        self.rnn = networks.CoreNetwork(256, 256)
        self.classifier = networks.ActionNetwork(hidden_size, 2)
        for i in range(self.agent_num):
            self.agents.append(Agent(h_g, h_l, glimpse_size, c, hidden_size, loc_dim, std, device))
    
    def reset(self):
        init_l_list = []
        for i in range(self.agent_num):
            #init_l = torch.FloatTensor(self.batch_size, 2).uniform_(-1.0,1.0).to(self.device)
            init_l = (-2 * torch.rand(2, 2) + 1).to(self.device)
            init_l.requires_grad = True
            init_l_list.append(init_l)
        init_h = torch.zeros(self.batch_size, 256, dtype=torch.float32, device=self.device, requires_grad=True) #lstm size

        return init_l_list, init_h

    def ram_loop(self, img, h_t, l_t, last=False):
        g_list, b_list, l_list, log_pi_list = [], [], [], []
        
        for i in range(self.agent_num):
            g_list.append(self.agents[i].glimpse_feature(img, l_t[i]))
        #g_list: len = agent_num, [b, hidden_size] in each element size
        
        s_t = self.selfatt(g_list)
        s_t = torch.unbind(s_t, dim=1) # s_t: agent_num*(batch_size, hidden_size)
        alpha, z_t = self.softatt(g_list, h_t)
        #z_t = torch.rand(2,256)
        #alpha = torch.rand(2,4)
        #h_t = self.lstm(z_t)
        h_t = self.rnn(z_t, h_t)
        #tempG = torch.cat(g_list, dim=1)
        #tempG: (batch_size, agent_num*hidden_size)
        
        for i in range(self.agent_num):
            log_pi, l_t = self.agents[i].location(s_t[i])
            b = self.agents[i].baseline(s_t[i])
            b_list.append(b)
            l_list.append(l_t)
            log_pi_list.append(log_pi)
        
        b_t = torch.stack(b_list, dim=1) #[agent_num, batch_size]
        log_pi_t = torch.stack(log_pi_list, dim=1)
        
        
        if last:
            log_probas = self.classifier(h_t)
            #log_probas = self.classifier(tempG)
            #log_probas: (batch_size, class_num)
            return l_list, b_t, log_pi_t, log_probas, alpha
        
        return h_t, l_list, b_t, log_pi_t


    def forward(self, img):
        l_t, h_t = self.reset()
        l_list, b_list, log_pi_list = [], [], []
        for i in range(self.glimpse_num-1):
            h_t, l_t, b_t, log_pi_t = self.ram_loop(img, h_t, l_t)
            l_list.append(l_t)
            b_list.append(b_t)
            log_pi_list.append(log_pi_t)
        l_t, b_t, log_pi_t, log_probas, alpha = self.ram_loop(img, h_t, l_t, last=True)
        
        l_list.append(l_t)
        b_list.append(b_t)
        log_pi_list.append(log_pi_t)

        return l_list, b_list, log_pi_list, log_probas, alpha

