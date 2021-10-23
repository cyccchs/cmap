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
        b, c, h, w = x.shape()
        start = self.denormalize(H, l)
        end = start + self.g

        x = F.pad(x, (self.g//2, self.g//2, self.g//2, self.g//2))

        patch = []
        for i in range(b):
            patch.append(x[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])

        return self.flatten(torch.stack(patch))
    
    def flatten(self, input_tensor)
        return torch.view(input_tensor[0], -1)
    
    def denormalize(self, T, coords):
        """
        T: the size of the image
        convert [-1, 1] to [0, T]
        """
        return(0.5 * ((coords + 1.0) * T)).long()

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
        
    def forward(self, x, l_pre):
        glimpse = self.retina.extract_patch(x, l_pre)
        l_prev = l_prev.view(l_prev.size(0), -1)
        
        g_out = F.relu(self.fc1(glimpse))
        l_out = F.relu(self.fc2(l_prev))

        what = self.fc3(g_out)
        where = self.fc4(l_out)

        g_t = F.relu(what + where)

        return g_t

