import torch
from torch.distributions import uniform
from networks import Retina, GlimpseNetwork, LocationNetwork, BaselineNetwork, CoreNetwork, SoftAtt


class Agent:
    def __init__(self, name, agent_num, ckpt_dir, h_g, h_l, k, s, glimpse_size, c, hidden_size, loc_dim, std, device):
        self.name = "agent_%s" % name
        self.ckpt_dir = ckpt_dir
        self.retina = GlimpseNetwork(self.name+"_retina.pth.tar", ckpt_dir, h_g, h_l, k, s, glimpse_size, c, device)
        self.location = LocationNetwork(self.name+"_loc.pth.tar", ckpt_dir, hidden_size, loc_dim, std, device)
        self.baseline = BaselineNetwork(self.name+"_base.pth.tar", ckpt_dir, hidden_size, 1, device)
        self.att = SoftAtt(self.name+"_att.pth.tar", ckpt_dir, hidden_size, device)
        self.lstm = CoreNetwork(self.name+"_lstm.pth.tar", ckpt_dir, hidden_size, device)
        self.train_param =[] 
        self.train_param.extend(list(self.retina.parameters()))
        self.train_param.extend(list(self.location.parameters()))
        self.train_param.extend(list(self.baseline.parameters()))
        self.train_param.extend(list(self.att.parameters()))
        self.train_param.extend(list(self.lstm.parameters()))

    def glimpse_feature(self,img, l_t):
        return self.retina(img, l_t)
    
    def location(self, s_t, sampling):
        return self.location(s_t, sampling)
    
    def baseline(self, s_t):
        return self.baseline(s_t)
    
    def save_model(self, is_best):
        self.retina.save_ckpt(is_best)
        self.location.save_ckpt(is_best)
        self.baseline.save_ckpt(is_best)
        self.att.save_ckpt(is_best)
        self.lstm.save_ckpt(is_best)
    
    def load_model(self, is_best):
        self.retina.load_ckpt(is_best)
        self.location.load_ckpt(is_best)
        self.baseline.load_ckpt(is_best)
        self.att.load_ckpt(is_best)
        self.lstm.load_ckpt(is_best)


        
