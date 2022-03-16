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

writer = SummaryWriter()
gpu = True
torch.autograd.set_detect_anomaly(True)
class train:
    def __init__(self):
        #self.ds = HRSC2016('./HRSC2016/Train/AllImages/image_names.txt')
        self.ds = HRSC2016('./HRSC2016/grayscale_test/AllImages/image_names.txt')
        self.collater = Collater(scales=800)
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.batch_size = 2
        self.glimpse_num = 6
        self.agent_num = 8
        self.epoch_num = 10000
        self.glimpse_size = 100
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
                    glimpse_size = self.glimpse_size,
                    glimpse_num = self.glimpse_num,
                    c = 3, 
                    lstm_size = 256, 
                    hidden_size = 256, 
                    loc_dim = 2, 
                    std = 0.22,
                    device = self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


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
            avg_loss, avg_rl, avg_act, avg_base, avg_acc, avg_reward = self.one_epoch(epoch)
            writer.add_scalar('avg acc', avg_acc, epoch)
            writer.add_scalar('avg rl loss', avg_rl, epoch)
            writer.add_scalar('avg action loss', avg_act, epoch)
            writer.add_scalar('avg basline loss', avg_base, epoch)
            writer.add_scalar('avg loss', avg_loss, epoch)
            writer.add_scalar('avg_reward', avg_reward, epoch)
            writer.flush()


    def one_epoch(self, epoch):

        self.model.train()
        print('EPOCH:', epoch)
        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        loss_rl, loss_act, loss_base = [], [], []
        with tqdm(enumerate(self.loader), total=len(self.loader)) as pbar:
            for j, (ni,batch) in enumerate(pbar):
                imgs, existence = batch['image'], batch['existence']
                imgs = imgs.to(self.device)
                existence = torch.tensor(existence).detach()
                existence = existence.to(self.device)
                self.optimizer.zero_grad()
                
                l_list, b_list, log_pi_list, log_probs, alpha = self.model(imgs)

                log_pi_all = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
                baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
                predicted = torch.max(log_probs, 1)[1]  #indices store in element[1]
                reward = (predicted.detach() == torch.tensor(existence).detach()).float()
                reward = self.weighted_reward(reward, alpha).to(self.device)
                
                draw(imgs, l_list, existence, predicted, reward, epoch, self.glimpse_size, self.agent_num)
                
                reward_list.append(torch.sum(reward)/len(reward))
                reward = reward.unsqueeze(1).repeat(1,self.glimpse_num,1) 
                #reward[batch_size, time_step, agent_num]
                advantage = reward - baselines.detach()
                
                loss_reinforce = torch.sum(-log_pi_all * advantage, dim=1)#sum along all glimpses
                #loss_reinforce = torch.sum(-log_pi_all * 1, dim=1)#sum along all glimpses
                
                loss_reinforce = torch.mean(loss_reinforce, dim=0).sum()
                #mean along all batch then sum up
                loss_baseline = F.mse_loss(baselines, reward)
                loss_action = F.nll_loss(log_probs, torch.tensor(existence))
                loss_reinforce = loss_reinforce*0.01
                loss = loss_action + loss_reinforce + loss_baseline
                #loss = loss_action + loss_reinforce*0.001
                
                correct = (predicted.detach()==torch.tensor(existence).detach()).float()
                acc = 100*(correct.sum()/len(existence))
                acc_list.append(acc)
                loss_list.append(loss)
                loss_rl.append(loss_reinforce)
                loss_act.append(loss_action)
                loss_base.append(loss_baseline)
                avg_loss = sum(loss_list)/len(loss_list)
                avg_rl = sum(loss_rl)/len(loss_rl)
                avg_act = sum(loss_act)/len(loss_act)
                avg_base = sum(loss_base)/len(loss_base)
                avg_acc = sum(acc_list)/len(acc_list)
                avg_reward = sum(reward_list)/len(reward_list)


                #writer.add_scalar('accuracy', acc, iteration)
                loss.backward()
                self.optimizer.step()
                iteration = iteration + 1
        return avg_loss, avg_rl, avg_act, avg_base, avg_acc, avg_reward
writer.close()

if __name__ == '__main__':
    trainer = train()
    trainer.train()






