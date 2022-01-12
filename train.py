import os
from dataloader import HRSC2016
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from collater_nobox import *
from model import *

ds = HRSC2016()
collater = Collater(scales=800)

loader = DataLoader(
    dataset=ds,
    batch_size=2,
    num_workers=8,
    collate_fn=collater,
    shuffle=False)

pbar = tqdm(enumerate(loader), total=len(loader))
model = RecurrentAttention(2, 128, 128, 3, 3, 256, 256, 2, 0.22)

def reset():
    init_l = torch.FloatTensor(2, 2).uniform_(-1.0,1.0) #batch size
    init_h = torch.zeros(2, 256, dtype=torch.float32, requires_grad=True) #lstm size
    
    return init_l, init_h
#def alpha_reward(reward, alpha):
#    for i in 

l_t, h_t = reset()

for i, (ni,batch) in enumerate(pbar):
    imgs, existence = batch['image'], batch['existence']
    l_list, b_list, log_pi_list = [], [], []
    R = 0
    for i in range(4-1):
        h_t, l_t, b_t, log_pi = model(imgs, h_t, l_t)
        l_list.append(l_t)
        b_list.append(b_t)
        log_pi_list.append(log_pi)
    
    h_t, l_t, b_t, log_pi, log_probs = model(imgs, h_t, l_t, last=True)
    l_list.append(l_t)
    b_list.append(b_t)
    log_pi_list.append(log_pi)
    
    #baselines = torch.stack(b_list).transpose(1,0)
    for i in b_list:
        torch.stack(i).transpose(1,0)
    log_pi = torch.stack(log_pi_list).transpose(1,0)
    predicted = torch.max(log_probs, 1)[1].detach()
    for i in range(len(existence)):
        R = R + (predicted[i] == existence[i])
