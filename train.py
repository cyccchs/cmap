import os
from dataloader import HRSC2016
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from collater_nobox import *
from model import *
import torch.nn.functional as F
import matplotlib.pyplot as plt


ds = HRSC2016()
collater = Collater(scales=800)
batch_size = 8 

loader = DataLoader(
    dataset=ds,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collater,
    shuffle=False)

model = MultiAgentRecurrentAttention(
            batch_size=batch_size, 
            h_g = 128,
            h_l = 128,
            glimpse_size = 10, 
            c = 3, 
            lstm_size = 256, 
            hidden_size = 256, 
            loc_dim = 2, 
            std = 0.22)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 1e-4 seems better now
#for name, para in model.named_parameters():
#    print(name)
#    print(para.shape)

def reset():
    
    init_l = torch.FloatTensor(batch_size, 2).uniform_(-1.0,1.0) #batch size
    init_l.requires_grad = True
    init_h = torch.zeros(batch_size, 256, dtype=torch.float32, requires_grad=True) #lstm size
    
    return init_l, init_h

def weighted_reward(batch_size, reward, alpha):
    
    reward_list = []
    for i in range(batch_size):
        if reward[i]==0:
            reward_list.append(4*alpha[i])
        else:
            reward_list.append(-4*alpha[i])
    
    return torch.stack(reward_list, dim=0)


l_t, h_t = reset()
torch.autograd.set_detect_anomaly(True)
pbar = tqdm(enumerate(loader), total=len(loader))

model.train()

accuracy = open('log.txt', 'w')
l = open('loss.txt', 'w')

for i, (ni,batch) in enumerate(pbar):
    
    imgs, existence = batch['image'], batch['existence']
    l_list, b_list, log_pi_list = [], [], []
    optimizer.zero_grad()
    
    for i in range(3):
        h_t, l_t, b_t, log_pi = model(imgs, h_t, l_t)
        l_list.append(l_t)
        b_list.append(b_t)
        log_pi_list.append(log_pi)
    
    h_t, l_t, b_t, log_pi, log_probs, alpha = model(imgs, h_t, l_t, last=True)
    l_list.append(l_t)
    b_list.append(b_t)
    log_pi_list.append(log_pi)

    
    log_pi_all = torch.stack(log_pi_list, dim=1).unsqueeze(2).repeat(1,1,4)
    baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
    predicted = torch.max(log_probs, 1)[1]  #indices store in element[1]
    print(predicted)
    print(torch.tensor(existence))
    reward = (predicted.detach() == torch.tensor(existence)).float()
    reward = weighted_reward(batch_size, reward, alpha)
    reward = reward.unsqueeze(1).repeat(1,4,1) #same shape as baselines
    advantage = reward - baselines.detach()
    
    loss_reinforce = torch.sum(-log_pi_all * advantage, dim=1)#sum along all glimpses
    loss_reinforce = torch.mean(loss_reinforce, dim=0).sum()#mea along all batch then sum up
    loss_baseline = F.mse_loss(baselines, reward)
    loss_action = F.nll_loss(log_probs, torch.tensor(existence))
    
    loss = loss_action + loss_baseline - loss_reinforce
    print('LOSS', loss)
    correct = (predicted==torch.tensor(existence)).float()
    acc = 100*(correct.sum()/len(existence))
    print('ACC', acc)
    loss.backward()
    optimizer.step()
    accuracy.write(str(acc.item()))
    accuracy.write("\n")
    l.write(str(loss.item()))
    l.write("\n")
#    plt.imshow(imgs[0].permute(1,2,0))
#    plt.show()
accuracy.close()
l.close()








