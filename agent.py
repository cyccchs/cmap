from network import Retina, GlimpseNetwork, LoactionNetwork, CoreNetwork, SelfAttention, SoftAttention

class Agent:
    def __init__(self):
        self.GlimpseNetwork = GlimpseNetwork()
        self.LocationNetwork = LocationNetwork()
        
