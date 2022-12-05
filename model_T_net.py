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
    
    def reset(self):
        #init loaction of all agents
        #init hidden state & cell state of lstm cell
        init_l_list, init_h_list, init_c_list = [], [], []
        
        for i in range(self.agent_num):
            init_l = torch.FloatTensor(self.batch_size, 2).uniform_(-1.0, 1.0).to(self.device)
            init_l.requires_grad = True
            init_l_list.append(init_l)

            init_h = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device, requires_grad=True)
            init_h_list.append(init_h)

            init_c = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device, requires_grad=True)
            init_c_list.append(init_c)

        return init_l_list, init_h_list, init_c_list

    def ram_loop(self, img, h_prev, c_prev, l_prev, i):
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
        if stop or i == self.glimpse_num - 1:
            last = True
        
        for j in range(self.agent_num):
            log_pi, l_t = self.agents[j].location(alpha_list[j])
            b = self.agents[j].baseline(alpha_list[j])
            b_list.append(b)
            l_list.append(l_t)
            log_pi_list.append(log_pi)
        
        b_t = torch.stack(b_list, dim=1) #[agent_num, batch_size]
        log_pi_t = torch.stack(log_pi_list, dim=1)
        
        log_prob = self.classifier(H_t) #[batch_size, class_num]
        
        return h_list, c_list, prob, l_list, b_t, log_pi_t, log_prob, alpha, last

    def forward(self, img):
        l_prev, h_prev, c_prev = self.reset()
        l_list, b_list, log_pi_list = [], [], []
        prob_list, terminate_list, log_prob_list = [], [], []
        prob = torch.zeros(1).to(self.device)
        T_reward = torch.zeros(1).to(self.device).detach()
        g_num = 0
        
        for i in range(self.glimpse_num):
            h_t, c_t, prob, l_t, b_t, log_pi_t, log_prob, alpha, last = self.ram_loop(img, h_prev, c_prev, l_prev, i)

            if last:
                terminate_list.append(torch.ones(1).to(self.device))
            else:
                terminate_list.append(torch.zeros(1).to(self.device))
            
            prob_list.append(prob)
            log_prob_list.append(log_prob)
            T_reward = T_reward - 0.5
            l_list.append(l_t)
            b_list.append(b_t)
            log_pi_list.append(log_pi_t)
            h_prev = h_t
            c_prev = c_t
            l_prev = l_t
            g_num = g_num + 1
            if last:
                break
        prob_ts = torch.stack(prob_list)
        log_probs = torch.stack(log_prob_list)
        terminate_ts = torch.stack(terminate_list)
                
        return  log_probs, prob_ts, terminate_ts, T_reward, l_list, b_list, log_pi_list, alpha, g_num
    
    def save_agent_ckpt(self, is_best=False):
        print("--------agents saving checkpoint in %s" % self.ckpt_dir)
        for agent in self.agents:
            agent.save_model(is_best)

    def load_agent_ckpt(self, is_best=False):
        print("--------agents loading checkpoint in %s" % self.ckpt_dir)
        for agent in self.agents:
            agent.load_model(is_best)


