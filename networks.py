import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


class Retina:
    
    def __init__(self, k, s, glimpse_size):
        self.g = glimpse_size
        self.k = k
        self.s = s
    
    def foveate(self, x, l):
        phi = []
        size = self.g
        for i in range (self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)
        phi = torch.cat(phi, 1)
        phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """
        x: (b,h,w,c) input image batch
        l: (b, 2) loaction
        size: glimpse size (int)
        """
        b, c, h, w = x.shape
        start = self.denormalize(h, l)
        end = start + size

        x = F.pad(x, (size//2+1, size//2+1, size//2+1, size//2+1))
        patch = []
        for i in range(b):
            patch.append(x[i, : , start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])
        return torch.stack(patch)

    def flatten(self, input_tensor):
        flat = []
        for i in range(input_tensor.shape[0]):
            flat.append(input_tensor[i].view(-1))
        return torch.stack(flat)

    def denormalize(self, T, coords):
        """
        T: the size of the image
        convert [-1, 1] to [0, T]
        """
        return (0.5 * ((coords + 1.0) * T)).long()

class GlimpseNetwork(nn.Module):
    """
        h_g: hidden layer for glimpse
        h_l: hidden layer for location
        k: num of patch
        s: scale factor of each subsequent patch (should be > 1)
        l_pre: location(l) of previous time step
    """
    
    def __init__(self, name, ckpt_dir, h_g, h_l, k, s, glimpse_size, c, device):
        super().__init__()
        self.ckpt_path = os.path.join(ckpt_dir, name)
        self.best_ckpt_path = os.path.join(ckpt_dir, "best_" + name)
        self.retina = Retina(k, s, glimpse_size)
        
        dimension = k * glimpse_size * glimpse_size * c
        self.fc1 = nn.Linear(dimension, h_g)
        dimension = 2
        self.fc2 = nn.Linear(dimension, h_l)


        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)
        self.to(device)

    def forward(self, x, l_prev):
        glimpse = self.retina.foveate(x, l_prev)
        l_prev = l_prev.view(l_prev.size(0), -1)
        
        g_out = F.relu(self.fc1(glimpse))
        l_out = F.relu(self.fc2(l_prev))

        what = self.fc3(g_out)
        where = self.fc4(l_out)

        g_t = F.relu(what + where)
        
        return g_t

    def save_ckpt(self, is_best):
        if is_best:
            torch.save(self.state_dict(), self.best_ckpt_path)
        else:
            torch.save(self.state_dict(), self.ckpt_path)
    def load_ckpt(self, is_best):
        if is_best:
            self.load_state_dict(torch.load(self.best_ckpt_path))
        else:
            self.load_state_dict(torch.load(self.ckpt_path))

class LocationNetwork(nn.Module):
    """
        input_size: input size of the fc layer
        output_size: output size of the fc layer
        std: standard deviation of the normal distribution
        h_t: hidden state vector of the core network for the
             current time step 't'

        mu: 2D vector of shape(B,2)
        l_t: 2D vector of shape(B,2)
    """
    
    def __init__(self, name, ckpt_dir, input_size, output_size, std, device):
        super().__init__()
        self.ckpt_path = os.path.join(ckpt_dir, name)
        self.best_ckpt_path = os.path.join(ckpt_dir, "best_" + name)

        self.std = std
        hidden_size = input_size // 2
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_lt = nn.Linear(input_size, output_size)
        self.to(device)

    def forward(self, s_t):
        #feat = F.relu(self.fc(s_t.detach()))
        #mu = torch.tanh(self.fc_lt(s_t.detach()))
        mu = torch.tanh(self.fc_lt(s_t))

        l_t = Normal(mu, self.std).rsample()
        #l_t = torch.tensor([[0.75, -0.75],[0.75, -0.75],[0.75, -0.75]])
        #l_t = torch.tensor([[0.75, -0.75],[0.75, -0.75]])
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t
    
    def save_ckpt(self, is_best):
        if is_best:
            torch.save(self.state_dict(), self.best_ckpt_path)
        else:
            torch.save(self.state_dict(), self.ckpt_path)
    def load_ckpt(self, is_best):
        if is_best:
            self.load_state_dict(torch.load(self.best_ckpt_path))
        else:
            self.load_state_dict(torch.load(self.ckpt_path))

class BaselineNetwork(nn.Module):
    """
        input_size: input_size of the fc layer
        output_size: output size of the fc layer
        h_t: hidden state vector of the core network for the current time step 't'

        b_t: 2D vector of shape (B, 1). The baseline for the current time step 't'
    """
    def __init__(self, name, ckpt_dir, input_size, output_size, device):
        super().__init__()
        self.ckpt_path = os.path.join(ckpt_dir, name)
        self.best_ckpt_path = os.path.join(ckpt_dir, "best_" + name)

        self.fc = nn.Linear(input_size, output_size)
        self.to(device)

    def forward(self, s_t):
        #b = self.fc(s_t.detach())
        b = self.fc(s_t)
        b = torch.squeeze(b)
        
        return b
    
    def save_ckpt(self, is_best):
        if is_best:
            torch.save(self.state_dict(), self.best_ckpt_path)
        else:
            torch.save(self.state_dict(), self.ckpt_path)
    def load_ckpt(self, is_best):
        if is_best:
            self.load_state_dict(torch.load(self.best_ckpt_path))
        else:
            self.load_state_dict(torch.load(self.ckpt_path))

class CoreNetwork(nn.Module): #CCI(LSTM cell)
    """
        h_t = relu( fc(h_t_prev) + fc(g_t))

        input_size: input size of the rnn
        hidden_size: hidden size of the rnn(256)
        g_t: 2D tensor of shape (B, hidden_size). Returned from glimpse network.
        h_prev: 2D tensor of shape (B, hidden_size). Hidden state for previous timestep.

        h_t: 2D tensor of shape (B, hidden_size). Hidden state for current timestep.
    """
    """ 
    def __init__(self, batch_size, lstm_size, device):
        super().__init__()

        self.lstm = nn.LSTMCell(lstm_size, lstm_size)
        self.h = torch.zeros(batch_size, lstm_size, dtype=torch.float32, requires_grad=True, device=device)
        self.c = torch.zeros(batch_size, lstm_size, dtype=torch.float32, requires_grad=True, device=device)
    def forward(self, z_t, h_t_prev):
         
        h_t, self.c = self.lstm(z_t, (h_t_prev,self.c))
        
        return h_t
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
    def forward(self, z_t, h_t_prev):
        h1 = self.i2h(z_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        
        return h_t

class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.size = hidden_size
        self.w = nn.Linear(hidden_size, hidden_size)
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, g_list):
        G = torch.stack(g_list, dim=1)
        G = G.detach()
        #G:[b, agent_num, hidden_size]
        x = self.w(G)
        q = self.wq(x)
        k = self.wk(x) #(b, agent_num, hidden_size)
        v = self.wv(x)
        q_trans = q.permute(0,2,1) #(b, hidden_size, agent_num)
        a_t = F.softmax(torch.matmul(k, q_trans)/self.size**0.5, dim=0)
        #matmul (b, agent_num, agent_num), softmax (b, agent_num, agent_num)
        s_t = torch.matmul(a_t, v)
        #s_t (b, agent_num, hidden_size)
        return s_t

class SoftAttention(nn.Module):
    
    def __init__(self, hidden_size, device):
        super().__init__()
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wg = nn.Linear(hidden_size, 1)
        self.device = device
    
    def forward(self, g_list, h_t):
        #G = torch.stack(g_list) # to represent 4 agents' g_t
        y_list = [torch.tanh(self.wk(g_list[i]) + self.wq(h_t)) for i in range(len(g_list))]
        m_list = [self.wg(y_list[i]) for i in range(len(g_list))]
        m_concat = torch.cat([m_list[i] for i in range(len(g_list))], dim=1)
        alpha = F.softmax(m_concat, dim=-1)
        z_list = [torch.mul(g_list[i], torch.index_select(alpha, 1, torch.tensor(i).to(self.device))) for i in range(len(g_list))]
        z_stack = torch.stack(z_list, 2)
        z_t = torch.sum(z_stack, 2) #similar to reduce_sum

        return alpha, z_t

class ActionNetwork(nn.Module):
    """The action network.
    Uses the internal state `h_t` of the core network to
    produce the final output classification.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.
    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.
    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.
    Returns:
        a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        action = F.log_softmax(self.fc(h_t), dim=1)
        
        return action

class TerminationNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, h_t):
        prob = self.fc(h_t.detach())
        prob = torch.sigmoid(prob)
        prob = torch.mean(prob)

        return prob
