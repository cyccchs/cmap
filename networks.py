
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
        b, c, h, w = x.shape()
        start = self.denormalize(H, l)
        end = start + self.g

        x = F.pad(x, (self.g//2, self.g//2, self.g//2, self.g//2))

        patch = []
        for i in range(b):
            patch.append(x[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])

        return self.flatten(torch.stack(patch))

    def flatten(self, input_tensor):
        return torch.view(input_tensor[0], -1)

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

        dimension = g * g * c
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
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

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

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())

        return b_t

class CoreNetwork(nn.Module): #CCI
    """
        h_t = relu( fc(h_t_prev) + fc(g_t))

        input_size: input size of the rnn
        hidden_size: hidden size of the rnni(256)
        g_t: 2D tensor of shape (B, hidden_size). Returned from glimpse network.
        h_prev: 2D tensor of shape (B, hidden_size). Hidden state for previous timestep.

        h_t: 2D tensor of shape (B, hidden_size). Hidden state for current timestep.
    """
    def __init__(self, input_size, hidden_size, glimpse_num):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = glimpse_num
        self.rnn = nn.LSTMCell(self.input_size,self.hidden_size)

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_prev)
        h_t = F.relu(h1 + h2)

        return h_t

class SelfAttention(nn.Module):
    def __init__(self, wg_size, wh_size):
        super().__init__()
        self.w = nn.Linear(256, 256, bias=False)
        self.wq = nn.Linear(256, 256, bias=False)
        self.wk = nn.Linear(256, 256, bias=False)
        self.wv = nn.Linear(256, 256, bias=False)
    def forward(self, g_ts):
        G = torch.stack(g_ts[0], g_ts[1], g_ts[2], g_ts[3])
        x = self.w(G)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q_trans = torch.transpose(q, 0, 1)
        a_t = F.softmax(torch.matmul(q_trans, k)/256**0.5)
        s_t = torch.matmul(a_t,v)

        return s_t

class SoftAttention(nn.Module):
    def __init__(self):
        self.wg = nn.Linear(256, 256, bias=False)
        self.wh = nn.Linear(256, 256, bias=False)
        self.wy = nn.Linear(256, 256)
    def forward(self, g_ts, h_prev):
        Y_t = torch.tanh(self.wg(g_ts) + self.wh(h_prev))
        alpha_t = F.softmax(self.wy(Y_t))
        z_t_list = [torch.mul(alpha_t[i], torch.index_select(g_ts, 1, torch.tensor(i))) for i in range(len(g_ts))]
        z_stack = torch.stack(z_t_list, 2)
        z_t = torch.sum(z_stack, 2) #similar to reduce_sum

        return alpha_t, z_t


