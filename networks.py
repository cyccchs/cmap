import os
import math
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
            torch.save(self.state_dict(), self.ckpt_path)
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
        self.fc_feat = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, g_t):
        feat = F.relu(self.fc_feat(g_t.detach()))
        mu = torch.tanh(self.fc_mu(feat))
        l_t = Normal(mu, self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)

        l_t = torch.clamp(l_t, -1, 1)
        
        return log_pi, l_t
    
    def save_ckpt(self, is_best):
        if is_best:
            torch.save(self.state_dict(), self.best_ckpt_path)
            torch.save(self.state_dict(), self.ckpt_path)
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

    def forward(self, alpha):
        b = self.fc(alpha.detach())
        b = torch.squeeze(b, 1)
        
        return b
    
    def save_ckpt(self, is_best):
        if is_best:
            torch.save(self.state_dict(), self.best_ckpt_path)
            torch.save(self.state_dict(), self.ckpt_path)
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
    
    def __init__(self, name, ckpt_dir, hidden_size, device):
        super().__init__()
        self.ckpt_path = os.path.join(ckpt_dir, name)
        self.best_ckpt_path = os.path.join(ckpt_dir, "best_" + name)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.to(device)

    def forward(self, z_t, h_prev, c_prev): 
        h_t, c_t = self.lstm(z_t, (h_prev, c_prev))
        
        return h_t, c_t
    
    def save_ckpt(self, is_best):
        if is_best:
            torch.save(self.state_dict(), self.best_ckpt_path)
            torch.save(self.state_dict(), self.ckpt_path)
        else:
            torch.save(self.state_dict(), self.ckpt_path)
    
    def load_ckpt(self, is_best):
        if is_best:
            self.load_state_dict(torch.load(self.best_ckpt_path))
        else:
            self.load_state_dict(torch.load(self.ckpt_path))

class SoftAtt(nn.Module):
    def __init__(self, name, ckpt_dir, hidden_size, device):
        super().__init__()
        self.ckpt_path = os.path.join(ckpt_dir, name)
        self.best_ckpt_path = os.path.join(ckpt_dir, "best_" + name)
        self.size = hidden_size
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)

        self.to(device)
    
    def forward(self, g_list, h_prev):
        G = torch.stack(g_list)
        q = self.wq(h_prev)
        k = self.wk(G)
        v = self.wv(G)

        att_score = torch.einsum('ijk,jk->ij', k, q)
        alpha = F.softmax(att_score, dim=0)
        h_t = torch.einsum('ij,ijk->jk', alpha, v)

        return alpha.permute(1,0), h_t
    
    def save_ckpt(self, is_best):
        if is_best:
            torch.save(self.state_dict(), self.best_ckpt_path)
            torch.save(self.state_dict(), self.ckpt_path)
        else:
            torch.save(self.state_dict(), self.ckpt_path)
    
    def load_ckpt(self, is_best):
        if is_best:
            self.load_state_dict(torch.load(self.best_ckpt_path))
        else:
            self.load_state_dict(torch.load(self.ckpt_path))

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
