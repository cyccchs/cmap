import torch
from torch.distributions import uniform
from networks import Retina, GlimpseNetwork, LocationNetwork, BaselineNetwork


class Agent:
    def __init__(self, h_g, h_l, glimpse_size, c, hidden_size, loc_dim, std, device):
        self.retina = GlimpseNetwork(h_g, h_l, glimpse_size, c, device)
        self.location = LocationNetwork(hidden_size, loc_dim, std, device)
        self.baseline = BaselineNetwork(hidden_size, 1, device)
    def glimpse_feature(self,img, l_t):
        return self.retina(img, l_t)
    def location(self, s_t):
        return self.location(s_t)
    def baseline(self, s_t):
        return self.baseline(s_t)

        
