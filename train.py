import os
import shutil
import time
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
is_train = True
class Trainer:
    def __init__(self):
        self.train_ds = HRSC2016('../HRSC2016/train/AllImages/image_names.txt')
        self.val_ds = HRSC2016('../HRSC2016/val/AllImages/image_names.txt')
        self.test_ds = HRSC2016('../HRSC2016/test/AllImages/image_names.txt')
        #self.ds = HRSC2016('./HRSC2016/grayscale_test/AllImages/image_names.txt')
        self.collater = Collater(scales=800)
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.draw_per_n_epoch = 1
        self.save_ckpt_per_n_epoch = 50
        self.resume = False
        self.best_val_acc = 0.0
        self.test_num = 25
        self.start_epoch = 0
        self.batch_size = 4
        self.glimpse_num = 2
        self.agent_num = 4
        self.epoch_num = 50000
        self.glimpse_size = 200
        self.ckpt_dir = "./ckpt"
        self.model_name = "{}agents_{}g_{}x{}".format(
                self.agent_num,
                self.glimpse_num,
                self.glimpse_size,
                self.glimpse_size
                )
        self.train_loader = DataLoader(
                    dataset=self.train_ds,
                    batch_size=self.batch_size,
                    num_workers=1,
                    collate_fn=self.collater,
                    shuffle=True,
                    drop_last=True)
        self.val_loader = DataLoader(
                    dataset=self.val_ds,
                    batch_size=self.batch_size,
                    num_workers=1,
                    collate_fn=self.collater,
                    shuffle=False,
                    drop_last=True)
        self.test_loader = DataLoader(
                    dataset=self.test_ds,
                    batch_size=self.batch_size,
                    num_workers=1,
                    collate_fn=self.collater,
                    shuffle=False,
                    drop_last=True)

        self.model = MultiAgentRecurrentAttention(
                    ckpt_dir = self.ckpt_dir,
                    batch_size=self.batch_size,
                    agent_num = self.agent_num,
                    h_g = 128,
                    h_l = 128,
                    glimpse_size = self.glimpse_size,
                    glimpse_num = self.glimpse_num,
                    c = 3, 
                    hidden_size = 256, 
                    loc_dim = 2, 
                    std = 0.2,
                    device = self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=100)


    def weighted_reward(self, reward, alpha):
        reward_list = []
        for i in range(self.batch_size):
            if reward[i]==0:
                reward_list.append(-1*alpha[i])
            else:
                reward_list.append(alpha[i])
        
        return torch.stack(reward_list, dim=0)

    def train(self):
        if self.resume:
            self.load_ckpt()
        for epoch in range(self.start_epoch, self.epoch_num):
            train_loss, train_rl, train_act, train_base, train_acc = self.train_one_epoch(epoch)
            val_loss, val_rl, val_act, val_base, val_acc = self.validate(epoch)
            is_best = val_acc > self.best_val_acc
            writer.add_scalar('train acc', train_acc, epoch)
            writer.add_scalar('train rl loss', train_rl, epoch)
            writer.add_scalar('train action loss', train_act, epoch)
            writer.add_scalar('train basline loss', train_base, epoch)
            writer.add_scalar('train loss', train_loss, epoch)
            writer.add_scalar('val acc', val_acc, epoch)
            writer.add_scalar('val rl loss', val_rl, epoch)
            writer.add_scalar('val action loss', val_act, epoch)
            writer.add_scalar('val basline loss', val_base, epoch)
            writer.add_scalar('val loss', val_loss, epoch)
            writer.flush()
            if epoch % self.save_ckpt_per_n_epoch == 0:
                self.save_ckpt(
						{
							"epoch": epoch +1,
							"model_state": self.model.state_dict(),
							"optim_state": self.optimizer.state_dict(),
						}, is_best)


    def train_one_epoch(self, epoch):
        self.model.train()
        print('EPOCH:', epoch)
        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        loss_rl, loss_act, loss_base = [], [], []
        start_t = time.time()
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader)) as pbar:
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
                reward = (predicted.detach() == existence).float()
                reward = self.weighted_reward(reward, alpha).to(self.device)
                if epoch % self.draw_per_n_epoch == 0 and iteration == 0:
                	draw(imgs, l_list, existence, predicted, self.batch_size, self.agent_num, self.glimpse_size, self.glimpse_num, epoch, 'train')
                
                reward_list.append(torch.sum(reward)/len(reward))
                reward = reward.unsqueeze(1).repeat(1,self.glimpse_num,1)
                #reward[batch_size, time_step, agent_num]
                advantage = reward - baselines.detach()
                
                loss_reinforce = torch.sum(-log_pi_all * advantage, dim=1)#sum along all glimpses
                #loss_reinforce = torch.sum(-log_pi_all * 1, dim=1)#sum along all glimpses
                
                loss_reinforce = torch.mean(loss_reinforce, dim=0).sum() #actor
                #mean along all batch then sum up
                loss_baseline = F.mse_loss(baselines, reward) #critic
                loss_action = F.nll_loss(log_probs, existence) #classification
                loss_reinforce = loss_reinforce*0.1
                loss = loss_action + loss_reinforce + loss_baseline
                #loss = loss_action + loss_reinforce*0.001
                
                correct = (predicted.detach() == existence).float()
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
                end_t = time.time()
                pbar.set_description(
                        ("{:.1f}s - loss: {:.2f} - acc: {:.2f}".format(
                            (end_t - start_t), loss.item(), avg_acc)
                        )
                )
                iteration = iteration + 1
        return avg_loss, avg_rl, avg_act, avg_base, avg_acc
    @torch.no_grad()
    def validate(self, epoch):
        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        loss_rl, loss_act, loss_base = [], [], []
        for i, batch in enumerate(self.val_loader):
            imgs, existence = batch['image'], batch['existence']
            imgs = imgs.to(self.device)
            existence = torch.tensor(existence).detach()
            existence = existence.to(self.device)
            
            l_list, b_list, log_pi_list, log_probs, alpha = self.model(imgs)

            log_pi_all = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
            baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
            predicted = torch.max(log_probs, 1)[1]  #indices store in element[1]
            reward = (predicted.detach() == existence).float()
            reward = self.weighted_reward(reward, alpha).to(self.device)
            
            if epoch % self.draw_per_n_epoch == 0:
                draw(imgs, l_list, existence, predicted, self.batch_size, self.agent_num, self.glimpse_size, self.glimpse_num, epoch, 'val', iteration)
            
            reward_list.append(torch.sum(reward)/len(reward))
            reward = reward.unsqueeze(1).repeat(1,self.glimpse_num,1)
            #reward[batch_size, time_step, agent_num]
            advantage = reward - baselines.detach()
            
            loss_reinforce = torch.sum(-log_pi_all * advantage, dim=1)#sum along all glimpses
            #loss_reinforce = torch.sum(-log_pi_all * 1, dim=1)#sum along all glimpses
            
            loss_reinforce = torch.mean(loss_reinforce, dim=0).sum() #actor
            #mean along all batch then sum up
            loss_baseline = F.mse_loss(baselines, reward) #critic
            loss_action = F.nll_loss(log_probs, existence) #classification
            loss_reinforce = loss_reinforce*0.1
            loss = loss_action + loss_reinforce + loss_baseline
            #loss = loss_action + loss_reinforce*0.001
            
            correct = (predicted.detach() == existence).float()
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

            iteration = iteration + 1
        return avg_loss, avg_rl, avg_act, avg_base, avg_acc
    
    @torch.no_grad()
    def test(self):
        correct = 0
        is_test = True
        self.load_ckpt()
        for i, batch in enumerate(self.test_loader):
            imgs, existence = batch['image'], batch['existence']
            imgs = imgs.to(self.device)
            existence = torch.tensor(existence).detach()
            existence = existence.to(self.device)
            
            l_list, b_list, log_pi_list, log_probs, alpha = self.model(imgs)

            pred = torch.max(log_probs, 1)[1]  #indices store in element[1]
            
            correct += (pred.clone().detach() == existence.clone().detach()).sum()
            
            draw(imgs, l_list, existence, pred, i, self.glimpse_size, self.agent_num)

        acc = 100.0*(correct/self.test_num)
        print("----test acc: {}/{} ({:.2f}%)".format(correct, self.test_num, acc))
    
    def save_ckpt(self, state, is_best=False):
        print("----Saving model in {}".format(self.ckpt_dir))
        filename = self.model_name + ".pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        self.model.save_agent_ckpt(is_best)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_ckpt(self, is_best=False):
        print("----Loading model from {}".format(self.ckpt_dir))
        filename = self.model_name + ".pth.tar"
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        self.start_epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.model.load_agent_ckpt(is_best)

        print("----Loaded {} checkpoint at epoch {}".format(filename, ckpt["epoch"]))


writer.close()
    
if __name__ == '__main__':
    trainer = Trainer()
    if is_train:
        trainer.train()
    else:
        trainer.test()






