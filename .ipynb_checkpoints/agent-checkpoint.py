import torch
from torch.distributions import uniform
from network import Retina, GlimpseNetwork, LoactionNetwork


class Agent:
    def __init__(self, h_g, h_l, glimpse_size, c, input_size, output_size, std):
        self.GlimpseNetwork = GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.LocationNetwork = LocationNetwork(input_size, output_size, std)
        self.BaselineNetwork = BaselineNetwork(input_size, output_size)
        self.init_location = uniform.Uniform(-1.0, 1.0).sample([16,2])
        #sample([batch_size, loc_dim])
    def glimpse_feature():
        return self.GlimpseNetwork(x, l_prev)
    def choose_action(self, x, l_prev):
        return self.LocationNetwork(h_t)
    def baseline_feedback():
        return self.BaselineNetwork(h_t)

        
