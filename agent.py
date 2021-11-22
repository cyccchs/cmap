import torch
from torch.distributions import uniform
from network import Retina, GlimpseNetwork, LoactionNetwork


class Agent:
    def __init__(self, h_g, h_l, glimpse_size, c, input_size, output_size, std):
        self.GlimpseNetwork = GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.LocationNetwork = LocationNetwork(input_size, output_size, std)
        self.init_location = uniform.Uniform(-1.0, 1.0).sample([16,2])
        #sample([batch_size, loc_dim])
    def choose_action(self, x, l_pre):
        g_t = self.GlimpseNetwork.forward(x, l_pre)
        log_pi, l_t = self.LocationNetwork(h_t)
        

    def save_model(self):
        pass
    def load_model(self):
        pass

        
