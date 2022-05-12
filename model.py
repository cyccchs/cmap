import torch.nn as nn
import torch
import networks
from agent import Agent

debug = False

class MultiAgentRecurrentAttention(nn.Module):
    def __init__(self, ckpt_dir, batch_size, agent_num, h_g, h_l, k, s, glimpse_size, glimpse_num, c, hidden_size, loc_dim, std, device):
        super().__init__()
        self.agents = []
        self.ckpt_dir = ckpt_dir
        self.agent_num = agent_num
        self.glimpse_num = glimpse_num
        self.batch_size = batch_size
        self.device = device
        self.selfatt = networks.SelfAttention(hidden_size)
        self.softatt = networks.SoftAttention(hidden_size, device)
        self.rnn = networks.CoreNetwork(hidden_size, hidden_size)
        self.classifier = networks.ActionNetwork(hidden_size, 10)
        self.termination = networks.TerminationNetwork(hidden_size)
        for i in range(self.agent_num):
            self.agents.append(Agent(i, ckpt_dir, h_g, h_l, k, s, glimpse_size, c, hidden_size, loc_dim, std, device))
    
    def reset(self):
        init_l_list = []
        for i in range(self.agent_num):
            init_l = torch.FloatTensor(self.batch_size, 2).uniform_(-1.0, 1.0).to(self.device)
            init_l.requires_grad = True
            init_l_list.append(init_l)
        init_h = torch.zeros(self.batch_size, 256, dtype=torch.float32, device=self.device, requires_grad=True) #lstm size

        return init_l_list, init_h

    def ram_loop(self, img, h_t, l_t, count, last=False):
        g_list, b_list, l_list, log_pi_list = [], [], [], []
        log_probas = -1
        for i in range(self.agent_num):
            g_list.append(self.agents[i].glimpse_feature(img, l_t[i]))
        #g_list: len = agent_num, [b, hidden_size] in each element size
        
        s_t = self.selfatt(g_list)
        s_t = torch.unbind(s_t, dim=1) # s_t: agent_num*(batch_size, hidden_size)
        alpha, z_t = self.softatt(g_list, h_t)
        #h_t = self.lstm(z_t, h_t)
        h_t = self.rnn(z_t, h_t)
        prob = self.termination(h_t)
        if prob.cpu().detach().numpy() > torch.rand(1).numpy() or count == self.glimpse_num - 1:
            last = True
        
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
            #log_probas: (batch_size, class_num)
            #return l_list, b_t, log_pi_t, log_probas, alpha
        
        return h_t, prob, l_list, b_t, log_pi_t, log_probas, alpha, last


    def forward(self, img):
        l_t, h_t = self.reset()
        l_list, b_list, log_pi_list = [], [], []
        prob_list, terminate_list, T_reward_list= [], [], []
        l_list.append(l_t)
        prob = torch.zeros(1)
        prob = prob.to(self.device)
        g_num = 0
        for i in range(self.glimpse_num):
            h_t, prob, l_t, b_t, log_pi_t, log_probas, alpha, last = self.ram_loop(img, h_t, l_t, i)
            prob_list.append(prob)
            if last:
                terminate_list.append(torch.ones(1).to(self.device))
            else:
                terminate_list.append(torch.zeros(1).to(self.device))
            T_reward_list.append(torch.tensor(-g_num, dtype=torch.float32, device=self.device))
            l_list.append(l_t)
            b_list.append(b_t)
            log_pi_list.append(log_pi_t)
            g_num = g_num + 1
            if last:
                prob_ts = torch.stack(prob_list)
                terminate_ts = torch.stack(terminate_list)
                T_reward_ts = torch.stack(T_reward_list)
                break
                
        return prob_ts, terminate_ts, T_reward_ts, l_list, b_list, log_pi_list, log_probas, alpha, g_num
        """
        l_t, b_t, log_pi_t, log_probas, alpha = self.ram_loop(img, h_t, l_t, last=True)
        
        prob_list.append(prob)
        terminate_list.append(torch.ones(1).to(self.device))
        T_reward_list.append(torch.tensor(-g_num, dtype=torch.float32, device=self.device))
        prob_ts = torch.stack(prob_list)
        terminate_ts = torch.stack(terminate_list)
        T_reward_ts = torch.stack(T_reward_list)
        l_list.append(l_t)
        b_list.append(b_t)
        log_pi_list.append(log_pi_t)
        g_num = g_num + 1
        print('g_num', g_num)

        return prob_ts, terminate_ts, T_reward_ts, l_list, b_list, log_pi_list, log_probas, alpha, g_num
        """
    def save_agent_ckpt(self, is_best=False):
        print("--------agents saving checkpoint in %s" % self.ckpt_dir)
        for agent in self.agents:
            agent.save_model(is_best)

    def load_agent_ckpt(self, is_best=False):
        print("--------agents loading checkpoint in %s" % self.ckpt_dir)
        for agent in self.agents:
            agent.load_model(is_best)


