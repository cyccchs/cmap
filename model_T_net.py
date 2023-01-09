import torch.nn as nn
import torch
import networks
from agent import Agent

class MultiAgentRecurrentAttention(nn.Module):
    def __init__(self, ckpt_dir, batch_size, agent_num, h_g, h_l, k, s, glimpse_size, glimpse_num, c, hidden_size, loc_dim, std, device):
        super().__init__()
        self.agents = []
        self.ckpt_dir = ckpt_dir
        self.batch_size = batch_size
        self.agent_num = agent_num
        self.glimpse_num = glimpse_num
        self.hidden_size = hidden_size
        self.device = device
        self.softatt = networks.SoftAttention(hidden_size, device)
        self.classifier = networks.ActionNetwork(hidden_size, 10)
        self.termination = networks.TerminationNetwork(hidden_size)
        for i in range(self.agent_num):
            self.agents.append(Agent(i, self.agent_num, ckpt_dir, h_g, h_l, k, s, glimpse_size, c, hidden_size, loc_dim, std, device))
    

    def forward(self, img, h_prev, c_prev, l_prev, i):
        g_list, b_list, l_list, alpha_list, h_list, c_list, log_pi_list = [], [], [], [], [], [], []
        log_prob = -1
        last = False
        stop = False
        
        for j in range(self.agent_num):
            g_list.append(self.agents[j].glimpse_feature(img, l_prev[j]))
            #g_list: agent_num*[b, hidden_size]
        
        #s_t = self.selfatt(g_list)
        #s_t = torch.unbind(s_t, dim=1)  #s_t: agent_num*[batch_size, hidden_size]
        for j in range(self.agent_num):
            alpha, z_t = self.softatt(g_list, h_prev[j])
            h_t, c_t = self.agents[j].lstm(z_t, h_prev[j], c_prev[j])
            h_list.append(h_t)
            c_list.append(c_t)
            alpha_list.append(alpha)

        H_t = torch.stack(h_list)
        prob = self.termination(H_t)
        stop = prob.cpu().detach().numpy() > torch.rand(1).numpy()
        
        #if i == self.glimpse_num - 1:
        if stop or i == self.glimpse_num - 1:
            last = True
        
        for j in range(self.agent_num):
            log_pi, l_t = self.agents[j].location(g_list[j])
            b = self.agents[j].baseline(g_list[j])
            b_list.append(b)
            l_list.append(l_t)
            log_pi_list.append(log_pi)
        
        b_t = torch.stack(b_list, dim=1) #[agent_num, batch_size]
        log_pi_t = torch.stack(log_pi_list, dim=1)
        
        log_prob = self.classifier(H_t) #[batch_size, class_num]

        return h_list, c_list, prob, l_list, b_t, log_pi_t, log_prob, alpha_list, last
    
    def save_agent_ckpt(self, is_best=False):
        print("--------agents saving checkpoint in %s" % self.ckpt_dir)
        for agent in self.agents:
            agent.save_model(is_best)

    def load_agent_ckpt(self, is_best=False):
        print("--------agents loading checkpoint in %s" % self.ckpt_dir)
        for agent in self.agents:
            agent.load_model(is_best)

