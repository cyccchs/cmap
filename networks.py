import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


class Retina:
    
    def __init__(self, glimpse_size):
        self.g = glimpse_size
    
    def extract_patch(self, x, l):
        """
        x: (b,h,w,c) input image batch
        l: (b, 2) loaction
        size: glimpse size (int)
        """
        b, c, h, w = x.shape
        start = self.denormalize(h, l)
        end = start + self.g

        x = F.pad(x, (self.g//2, self.g//2, self.g//2, self.g//2))
        patch = []
        for i in range(b):
            patch.append(x[i, : , start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])
        
        return self.flatten(torch.stack(patch))

    def flatten(self, input_tensor):
        return input_tensor[0].view(-1)

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
        l_pre: location(l) of previous time step
        """
    
    def __init__(self, h_g, h_l, glimpse_size, c):
        super().__init__()
        self.retina = Retina(glimpse_size)

        dimension = glimpse_size * glimpse_size * c
        self.fc1 = nn.Linear(dimension, h_g)

        dimension = 2
        self.fc2 = nn.Linear(dimension, h_l)

        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_prev):
        glimpse = self.retina.extract_patch(x, l_prev)
        l_prev = l_prev.view(l_prev.size(0), -1)

        g_out = F.relu(self.fc1(glimpse))
        l_out = F.relu(self.fc2(l_prev))

        what = self.fc3(g_out)
        where = self.fc4(l_out)

        g_t = F.relu(what + where)
        
        return g_t

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
    
    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std
        self.hidden_size = input_size//2
        self.fc = nn.Linear(input_size, self.hidden_size)
        self.fc_lt = nn.Linear(self.hidden_size, output_size)

    def forward(self, h_t):
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))

        l_t = Normal(mu, self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)

        log_pi = torch.sum(log_pi, dim=1)
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t

class BaselineNetwork(nn.Module):
    """
        input_size: input_size of the fc layer
        output_size: output size of the fc layer
        h_t: hidden state vector of the core network for the current time step 't'

        b_t: 2D vector of shape (B, 1). The baseline for the current time step 't'
    """
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, s_t):
        b = self.fc(s_t.detach())
        b = torch.squeeze(b)
        
        return b

class CoreNetwork(nn.Module): #CCI(LSTM cell)
    """
        h_t = relu( fc(h_t_prev) + fc(g_t))

        input_size: input size of the rnn
        hidden_size: hidden size of the rnn(256)
        g_t: 2D tensor of shape (B, hidden_size). Returned from glimpse network.
        h_prev: 2D tensor of shape (B, hidden_size). Hidden state for previous timestep.

        h_t: 2D tensor of shape (B, hidden_size). Hidden state for current timestep.
    """
    def __init__(self, batch_size, lstm_size):
        super().__init__()

        self.lstm = nn.LSTMCell(lstm_size, lstm_size)
        self.h = torch.zeros(batch_size, lstm_size, dtype=torch.float32, requires_grad=True)
        self.c = torch.zeros(batch_size, lstm_size, dtype=torch.float32, requires_grad=True)
    def forward(self, z_t):
        
        self.h, self.c = self.lstm(z_t, (self.h,self.c))
        
        return self.h.detach()

class SelfAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(256, 256)
        self.wq = nn.Linear(256, 256)
        self.wk = nn.Linear(256, 256)
        self.wv = nn.Linear(256, 256)
    
    def forward(self, g_list):
        G = torch.stack(g_list, dim=1)
        x = self.w(G)
        q = self.wq(x)
        k = self.wk(x) #(b, 4, 256)
        v = self.wv(x)
        q_trans = torch.transpose(q, 1, 2) #(b, 256, 4)
        a_t = F.softmax(torch.matmul(k, q_trans)/256**0.5, dim=-1)
        #matmul (b, 4, 4), softmax (b, 4, 4)
        s_t = torch.matmul(a_t, v)

        return s_t

class SoftAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.wk = nn.Linear(256, 256)
        self.wq = nn.Linear(256, 256)
        self.wg = nn.Linear(256, 1)
    
    def forward(self, g_list, h_t):
        G = torch.stack(g_list) # to represent 4 agents' g_t
        y_list = [torch.tanh(self.wk(G[i]) + self.wq(h_t)) for i in range(len(G))]
        m_list = [self.wg(y_list[i]) for i in range(len(G))]
        m_concat = torch.cat([m_list[i] for i in range(len(G))], dim=1)
        alpha = F.softmax(m_concat, dim=-1)
        z_list = [torch.mul(G[i], torch.index_select(alpha, 1, torch.tensor(i))) for i in range(len(G))]
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
