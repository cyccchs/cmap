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
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

writer = SummaryWriter()
gpu = True
is_train = True
class Trainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.train_ds = torchvision.datasets.CIFAR10(
            root = '../',
            train = is_train,
            transform = self.transform,
            download = True
        )
        self.val_ds = torchvision.datasets.CIFAR10(
            root = '../',
            train = False,
            transform = self.transform,
            download = True
        )
        self.collater = Collater(scales=32)
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.draw_per_n_epoch = 50
        self.save_ckpt_per_n_epoch = 50
        self.train_terminate = True
        self.resume = True
        self.duration = 99999
        self.save_gap = 300
        self.epoch_time = 0.0
        self.best_val_acc = 0.0
        self.test_num = 25
        self.start_epoch = 0
        self.batch_size = 2
        self.glimpse_size = 7
        self.scale = 2.0
        self.patch_num = 2
        self.glimpse_num = 8
        self.agent_num = 2
        self.epoch_num = 50000
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
                    shuffle=True,
                    drop_last=True)
        self.val_loader = DataLoader(
                    dataset=self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=True)

        self.model = MultiAgentRecurrentAttention(
                    ckpt_dir = self.ckpt_dir,
                    batch_size = self.batch_size,
                    agent_num = self.agent_num,
                    h_g = 128,
                    h_l = 128,
                    k = self.patch_num,
                    s = self.scale,
                    glimpse_size = self.glimpse_size,
                    glimpse_num = self.glimpse_num,
                    c = 3, 
                    hidden_size = 256, 
                    loc_dim = 2, 
                    std = 0.2,
                    device = self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", verbose=True, min_lr=1e-5, patience=50)


    def weighted_reward(self, reward, alpha):
        reward_list = []
        for i in range(self.batch_size):
            if reward[i]==0:
                reward_list.append(-4*alpha[i])
            else:
                reward_list.append(4*alpha[i])
        
        return torch.stack(reward_list, dim=0)
    def terminate_reward_function(self, reward_ts, reward, g_num):
        if torch.mean(reward) > 0.55:
            terminate_reward = (reward_ts + 5) / g_num
        else:
            terminate_reward = (reward_ts - 5) / g_num
        return terminate_reward

    def train(self):
        if self.resume:
            self.load_ckpt(is_best=True)
        for epoch in range(self.start_epoch, self.epoch_num):
            train_loss, train_rl, train_act, train_base, train_T, train_g_num, train_acc = self.train_one_epoch(epoch)
            val_loss, val_rl, val_act, val_base, val_T, val_g_num, val_acc = self.validate(epoch)
            if val_acc > self.best_val_acc:
                is_best = True
                self.best_val_acc = val_acc.item()
            self.scheduler.step(-val_acc)
            writer.add_scalar('train acc', train_acc, epoch)
            writer.add_scalar('train glimpse number', train_g_num, epoch)
            writer.add_scalar('train rl loss', train_rl, epoch)
            writer.add_scalar('train action loss', train_act, epoch)
            writer.add_scalar('train basline loss', train_base, epoch)
            writer.add_scalar('train terminate loss', train_T, epoch)
            writer.add_scalar('train loss', train_loss, epoch)
            writer.add_scalar('val acc', val_acc, epoch)
            writer.add_scalar('val rl loss', val_rl, epoch)
            writer.add_scalar('val action loss', val_act, epoch)
            writer.add_scalar('val basline loss', val_base, epoch)
            writer.add_scalar('val terminate loss', val_T, epoch)
            writer.add_scalar('val glimpse number', val_g_num, epoch)
            writer.add_scalar('val loss', val_loss, epoch)
            writer.flush()
            #if epoch % self.save_ckpt_per_n_epoch == 0:
            if self.duration > self.save_gap:
                self.duration = 0
                self.save_ckpt(
						{
							"epoch": epoch +1,
							"model_state": self.model.state_dict(),
							"optim_state": self.optimizer.state_dict(),
						}, is_best)
            self.duration = self.duration + self.epoch_time

    def train_one_epoch(self, epoch):
        self.model.train()
        print('EPOCH:', epoch, end=' / ')
        print('best val_acc: {:.2f}'.format(self.best_val_acc), end=' / ')
        print('lr:', self.optimizer.param_groups[0]['lr'])
        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        loss_rl, loss_act, loss_base, loss_T = [], [], [], []
        g_num_list = []
        start_t = time.time()
        with tqdm(self.train_loader, total=len(self.train_loader)) as pbar:
            for j, (images, labels) in enumerate(pbar):
                imgs = images.to(self.device)
                existence = labels.to(self.device)
                self.optimizer.zero_grad()
                
                prob_ts, terminate_ts, T_reward_ts, l_list, b_list, log_pi_list, log_probs, alpha, g_num = self.model(imgs, self.train_terminate)

                log_pi_all = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
                baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
                predicted = torch.max(log_probs, 1)[1]  #indices store in element[1]
                reward = (predicted.detach() == existence).float()
                terminate_reward = self.terminate_reward_function(T_reward_ts, reward, g_num)
                reward = self.weighted_reward(reward, alpha).to(self.device)
                
                #if epoch % self.draw_per_n_epoch == 0 and iteration == 0:
                if self.duration > self.save_gap and iteration == 0:
                	draw(imgs, l_list, existence, predicted, self.batch_size, self.agent_num, self.glimpse_size, g_num, self.patch_num, self.scale, epoch, 'train')
                
                reward_list.append(torch.mean(reward))
                reward = reward.unsqueeze(1).repeat(1,g_num,1)
                #reward[batch_size, time_step, agent_num]
                adjusted_reward = reward - baselines.detach()
                
                loss_reinforce = torch.sum(-log_pi_all * adjusted_reward, dim=1)#sum along all glimpses
                #loss_reinforce = torch.sum(-log_pi_all * 1, dim=1)#sum along all glimpses
                
                loss_reinforce = torch.mean(loss_reinforce, dim=0).sum() #actor
                #mean along all batch then sum up
                loss_baseline = F.mse_loss(baselines, reward) #critic
                loss_action = F.nll_loss(log_probs, existence) #classification
                loss_reinforce = loss_reinforce*0.1
                prob_ts = torch.unsqueeze(prob_ts, 1)
                loss_terminate = F.binary_cross_entropy_with_logits(prob_ts, terminate_ts)
                loss_terminate = torch.mean(loss_terminate * terminate_reward)
                loss = loss_action + loss_reinforce + loss_baseline + loss_terminate
                #loss = loss_action + loss_reinforce*0.001
                
                correct = (predicted.detach() == existence).float()
                acc = 100*(correct.sum()/len(existence))
                acc_list.append(acc.detach())
                g_num_list.append(g_num)
                loss_list.append(loss.detach())
                loss_rl.append(loss_reinforce.detach())
                loss_act.append(loss_action.detach())
                loss_base.append(loss_baseline.detach())
                loss_T.append(loss_terminate.detach())
                avg_loss = sum(loss_list)/len(loss_list)
                avg_rl = sum(loss_rl)/len(loss_rl)
                avg_act = sum(loss_act)/len(loss_act)
                avg_base = sum(loss_base)/len(loss_base)
                avg_terminate = sum(loss_T)/len(loss_T)
                avg_g_num = sum(g_num_list)/len(g_num_list)
                avg_acc = sum(acc_list)/len(acc_list)
                avg_reward = sum(reward_list)/len(reward_list)


                #writer.add_scalar('accuracy', acc, iteration)
                loss.backward()
                self.optimizer.step()
                end_t = time.time()
                self.epoch_time = end_t - start_t
                pbar.set_description(
                        ("{:.1f}s - loss: {:.2f} - acc: {:.2f}".format(
                            self.epoch_time, avg_loss, avg_acc)
                        )
                )
                iteration = iteration + 1
        return avg_loss, avg_rl, avg_act, avg_base, avg_terminate, avg_g_num, avg_acc
    @torch.no_grad()
    def validate(self, epoch):
        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        loss_rl, loss_act, loss_base, loss_T = [], [], [], []
        g_num_list = []
        for i, (images, labels) in enumerate(self.val_loader):
            imgs = images.to(self.device)
            existence = labels.to(self.device)
            
            prob_ts, terminate_ts, T_reward_ts, l_list, b_list, log_pi_list, log_probs, alpha, g_num = self.model(imgs, self.train_terminate)

            log_pi_all = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
            baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
            predicted = torch.max(log_probs, 1)[1]  #indices store in element[1]
            reward = (predicted.detach() == existence).float()
            terminate_reward = self.terminate_reward_function(T_reward_ts, reward, g_num)
            reward = self.weighted_reward(reward, alpha).to(self.device)
            
            #if epoch % self.draw_per_n_epoch == 0 and iteration == 0:
            if self.duration > self.save_gap and iteration == 0:
                draw(imgs, l_list, existence, predicted, self.batch_size, self.agent_num, self.glimpse_size, g_num, self.patch_num, self.scale, epoch, 'val', iteration)
            
            reward_list.append(torch.mean(reward))
            reward = reward.unsqueeze(1).repeat(1,g_num,1)
            #reward[batch_size, time_step, agent_num]
            adjusted_reward = reward - baselines.detach()
            
            loss_reinforce = torch.sum(-log_pi_all * adjusted_reward, dim=1)#sum along all glimpses
            #loss_reinforce = torch.sum(-log_pi_all * 1, dim=1)#sum along all glimpses
            
            loss_reinforce = torch.mean(loss_reinforce, dim=0).sum() #actor
            #mean along all batch then sum up
            loss_baseline = F.mse_loss(baselines, reward) #critic
            loss_action = F.nll_loss(log_probs, existence) #classification
            loss_reinforce = loss_reinforce*0.1
            prob_ts = torch.unsqueeze(prob_ts, 1)
            loss_terminate = F.binary_cross_entropy_with_logits(prob_ts, terminate_ts)
            loss_terminate = torch.mean(loss_terminate * terminate_reward)
            loss = loss_action + loss_reinforce + loss_baseline + loss_terminate
            #loss = loss_action + loss_reinforce*0.001
            
            correct = (predicted.detach() == existence).float()
            acc = 100*(correct.sum()/len(existence))
            acc_list.append(acc.detach())
            g_num_list.append(g_num)
            loss_list.append(loss.detach())
            loss_rl.append(loss_reinforce.detach())
            loss_act.append(loss_action.detach())
            loss_base.append(loss_baseline.detach())
            loss_T.append(loss_terminate.detach())
            avg_loss = sum(loss_list)/len(loss_list)
            avg_rl = sum(loss_rl)/len(loss_rl)
            avg_act = sum(loss_act)/len(loss_act)
            avg_base = sum(loss_base)/len(loss_base)
            avg_terminate = sum(loss_T)/len(loss_T)
            avg_g_num = sum(g_num_list)/len(g_num_list)
            avg_acc = sum(acc_list)/len(acc_list)
            avg_reward = sum(reward_list)/len(reward_list)
            iteration = iteration + 1
        return avg_loss, avg_rl, avg_act, avg_base, avg_terminate, avg_g_num, avg_acc
    
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






