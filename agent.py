from network import Retina, GlimpseNetwork, LoactionNetwork, CoreNetwork, SelfAttention, SoftAttention

class Agent:
    def __init__(self, h_g, h_l, glimpse_size, c, input_size, output_size, std):
        self.GlimpseNetwork = GlimpseNetwork(h_g, h_l, glimpse_size, c)
        self.LocationNetwork = LocationNetwork(input_size, output_size, std)
    def next_step(self, x, l_pre, h_t):
        pass

    def save_model(self):
        pass
    def load_model(self):
        pass

        
