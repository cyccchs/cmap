import os
from dataloader import HRSC2016
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from collater_nobox import *
from model import *
from utils import draw
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)
writer = SummaryWriter()

class train:
    def __init__(self):
        self.ds = HRSC2016('./HRSC2016/Train/AllImages/image_names.txt')
        self.collater = Collater(scales=800)
        self.batch_size = 2
        self.glimpse_num = 8
        self.agent_num = 4
        self.epoch_num = 10000
        self.loader = DataLoader(
                    dataset=self.ds,
                    batch_size=self.batch_size,
                    num_workers=1,
                    collate_fn=self.collater,
                    shuffle=True)

        self.model = MultiAgentRecurrentAttention(
                    batch_size=self.batch_size,
                    agent_num = self.agent_num,
                    h_g = 128,
                    h_l = 128,
                    glimpse_size = 50, 
                    c = 3, 
                    lstm_size = 256, 
                    hidden_size = 256, 
                    loc_dim = 2, 
                    std = 0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

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
                reward_list.append(-1*alpha[i])
            else:
                reward_list.append(alpha[i])
        
        return torch.stack(reward_list, dim=0)

    def train(self):
        for epoch in range(self.epoch_num):
            avg_acc, avg_loss, avg_reward = self.one_epoch(epoch)
            writer.add_scalar('avg acc', avg_acc, epoch)
            writer.add_scalar('avg loss', avg_loss, epoch)
            writer.add_scalar('avg_reward', avg_reward, epoch)


    def one_epoch(self, epoch):

        self.model.train()
        print('EPOCH:', epoch)
        l_t, h_t = self.reset()
        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        with tqdm(enumerate(self.loader), total=len(self.loader)) as pbar:
            for j, (ni,batch) in enumerate(pbar):
                imgs, existence = batch['image'], batch['existence']
                l_list, b_list, log_pi_list = [], [], []
                self.optimizer.zero_grad()
                
                for i in range(self.glimpse_num - 1):
                    h_t, l_t, b_t, log_pi_t = self.model(imgs, h_t, l_t)
                    l_list.append(l_t)
                    b_list.append(b_t)
                    log_pi_list.append(log_pi_t)
                
                #h_t, l_t, b_t, log_pi, log_probs, alpha = self.model(imgs, h_t, l_t, last=True)
                h_t, l_t, b_t, log_pi, log_probs = self.model(imgs, h_t, l_t, last=True)
                alpha = torch.ones(self.batch_size, self.agent_num, dtype=torch.float32)
                l_list.append(l_t)
                #l_list (glimpse_num, agent_num, [batch_size, location])
                """
                print('l list', len(l_list))
                print('l list[0]', len(l_list[0]))
                print('l', l_list[0][0].shape)
                print('imgs', imgs[0].shape)
                print(imgs[0][0][0])
                """ 
                b_list.append(b_t)
                log_pi_list.append(log_pi_t)

                log_pi_all = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
                baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
                predicted = torch.max(log_probs, 1)[1]  #indices store in element[1]
                reward = (predicted.detach() == torch.tensor(existence)).float()
                
                reward = self.weighted_reward(reward, alpha)
                if epoch%2 == 0:
                    draw(imgs, l_list, existence, predicted.detach(), reward, epoch)
                
                reward_list.append(torch.sum(reward)/len(reward))
                reward = reward.unsqueeze(1).repeat(1,self.glimpse_num,1) 
                    #[batch_size, time_step, agent_num]
                advantage = reward - baselines.detach()
                
                loss_reinforce = torch.sum(-log_pi_all * advantage, dim=1)#sum along all glimpses
                #loss_reinforce = torch.sum(-log_pi_all * 1, dim=1)#sum along all glimpses
                
                loss_reinforce = torch.mean(loss_reinforce, dim=0).sum()#mean along all batch then sum up
                loss_baseline = F.mse_loss(baselines, reward)
                loss_action = F.nll_loss(log_probs, torch.tensor(existence))
                
                loss = loss_action - loss_reinforce + loss_baseline
                writer.add_scalar('action loss', loss_action.item(), iteration)
                writer.add_scalar('baseline loss', loss_baseline.item(), iteration)
                writer.add_scalar('reinforce loss', loss_reinforce.item(), iteration)
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
                #print('ACC', acc)
                loss.backward()
                self.optimizer.step()
                iteration = iteration + 1
                writer.flush()
        return sum(acc_list)/len(acc_list), sum(loss_list)/len(loss_list), sum(reward_list)/len(reward_list)           
writer.close()

if __name__ == '__main__':
    trainer = train()
    trainer.train()






