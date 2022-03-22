import torch
from torch.distributions import uniform
from networks import Retina, GlimpseNetwork, LocationNetwork, BaselineNetwork


class Agent:
    def __init__(self, name, ckpt_dir, h_g, h_l, glimpse_size, c, hidden_size, loc_dim, std, device):
        self.name = "agent_%s" % name
        self.ckpt_dir = ckpt_dir
        self.retina = GlimpseNetwork(self.name+"_retina.pth.tar", ckpt_dir, h_g, h_l, glimpse_size, c, device)
        self.location = LocationNetwork(self.name+"_loc.pth.tar", ckpt_dir, hidden_size, loc_dim, std, device)
        self.baseline = BaselineNetwork(self.name+"_base.pth.tar", ckpt_dir, hidden_size, 1, device)

    def glimpse_feature(self,img, l_t):
        return self.retina(img, l_t)
    
    def location(self, s_t):
        return self.location(s_t)
    
    def baseline(self, s_t):
        return self.baseline(s_t)
    
    def save_model(self):
        self.retina.save_ckpt()
        self.location.save_ckpt()
        self.baseline.save_ckpt()
    
    def load_model(self):
        self.retina.load_ckpt()
        self.location.load_ckpt()
        self.baseline.load_ckpt()


        
