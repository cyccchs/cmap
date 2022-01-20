import os
from dataloader import HRSC2016
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from collater_nobox import *
from model import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)
writer = SummaryWriter()

class train:
    def __init__(self):
        self.ds = HRSC2016('./HRSC2016/FullDataSet/AllImages/image_names.txt')
        self.collater = Collater(scales=800)
        self.batch_size = 16
        self.glimpse_num = 4
        self.agent_num = 4
        self.epoch_num = 100
        self.loader = DataLoader(
                    dataset=self.ds,
                    batch_size=self.batch_size,
                    num_workers=8,
                    collate_fn=self.collater,
                    shuffle=False)

        self.model = MultiAgentRecurrentAttention(
                    batch_size=self.batch_size,
                    agent_num = self.agent_num,
                    h_g = 128,
                    h_l = 128,
                    glimpse_size = 9, 
                    c = 3, 
                    lstm_size = 256, 
                    hidden_size = 256, 
                    loc_dim = 2, 
                    std = 0.8)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

    def reset(self):
        
        init_l_list = []
        for i in range(self.agent_num):
            init_l = torch.FloatTensor(self.batch_size, 2).uniform_(-1.0,1.0) #batch size
            init_l.requires_grad = True
            init_l_list.append(init_l)
        init_h = torch.zeros(self.batch_size, 256, dtype=torch.float32, requires_grad=True) #lstm size
        
        return init_l_list, init_h

    def weighted_reward(self, reward, alpha):
        reward_list = []
        for i in range(self.batch_size):
            if reward[i]==0:
                reward_list.append(4*alpha[i])
            else:
                reward_list.append(-4*alpha[i])
        
        return torch.stack(reward_list, dim=0)

    def train(self):
        for epoch in range(self.epoch_num):
            avg_acc, avg_loss = self.one_epoch(epoch)
            writer.add_scalar('avg acc', avg_acc, epoch)
            writer.add_scalar('avg loss', avg_loss, epoch)


    def one_epoch(self, epoch):

        self.model.train()
        print('EPOCH:', epoch)
        l_t, h_t = self.reset()
        iteration = 0
        acc_list, loss_list = [], []
        pbar = tqdm(enumerate(self.loader), total=len(self.loader))
        for j, (ni,batch) in enumerate(pbar):
            imgs, existence = batch['image'], batch['existence']
            l_list, b_list, log_pi_list = [], [], []
            self.optimizer.zero_grad()
            
            for i in range(self.glimpse_num - 1):
                print('glimpse num: ', i)
                h_t, l_t, b_t, log_pi_t = self.model(imgs, h_t, l_t)
                l_list.append(l_t)
                b_list.append(b_t)
                log_pi_list.append(log_pi_t)
            
            #h_t, l_t, b_t, log_pi, log_probs, alpha = self.model(imgs, h_t, l_t, last=True)
            h_t, l_t, b_t, log_pi, log_probs = self.model(imgs, h_t, l_t, last=True)
            alpha = torch.ones(16, 4, dtype=torch.float32)
            l_list.append(l_t)
            b_list.append(b_t)
            log_pi_list.append(log_pi_t)

            log_pi_all = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
            baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
            predicted = torch.max(log_probs, 1)[1]  #indices store in element[1]
            reward = (predicted.detach() == torch.tensor(existence)).float()
            reward = self.weighted_reward(reward, alpha)
            reward = reward.unsqueeze(1).repeat(1,self.glimpse_num,1) 
                #[batch_size, time_step, agent_num]
            advantage = reward - baselines.detach()
            
            loss_reinforce = torch.sum(-log_pi_all * advantage, dim=1)#sum along all glimpses
            loss_reinforce = torch.mean(loss_reinforce, dim=0).sum()#mean along all batch then sum up
            loss_baseline = F.mse_loss(baselines, reward)
            loss_action = F.nll_loss(log_probs, torch.tensor(existence))
            
            loss = loss_action + loss_baseline - loss_reinforce*0.1
            writer.add_scalar('action loss', loss_action.item(), iteration)
            writer.add_scalar('baseline loss', loss_baseline.item(), iteration)
            writer.add_scalar('reinforce loss', 0.1*loss_reinforce.item(), iteration)
            """
            print('LOSS: ', 
                    'action:', loss_action.item(), 
                    'baseline:', loss_baseline.item(), 
                    'r:', 0.1*loss_reinforce.item())
            print(predicted)
            print('LOSS: ', loss.item())
            """
            correct = (predicted==torch.tensor(existence)).float()
            acc = 100*(correct.sum()/len(existence))
            acc_list.append(acc)
            loss_list.append(loss)

            writer.add_scalar('accuracy', acc, iteration)
            print('ACC', acc)
            loss.backward()
            self.optimizer.step()
            iteration = iteration + 1
            writer.flush()
        return sum(acc_list)/len(acc_list), sum(loss_list)/len(loss_list)           
writer.close()

if __name__ == '__main__':
    trainer = train()
    trainer.train()






