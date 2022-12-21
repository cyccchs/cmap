import os
import shutil
import time
from dataloader import HRSC2016
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from collater_nobox import *
from model_T_net import *
from utils import draw, AvgMeter
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from scipy.stats import entropy
writer = SummaryWriter()
gpu = True
class Trainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.train_ds = torchvision.datasets.CIFAR10(
            root = '../',
            train = True,
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
        self.resume = False
        self.pbar_detail = False
        self.M = 10 #monte carlo sample = 10
        self.duration = 99999
        self.save_gap = 300
        self.epoch_time = 0.0
        self.best_val_acc = 0.0
        self.start_epoch = 0
        self.batch_size = 16
        self.glimpse_size = 8
        self.scale = 1.0
        self.patch_num = 1
        self.glimpse_num = 6
        self.agent_num = 4
        self.epoch_num = 400
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
                    batch_size = self.batch_size * self.M,
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
        self.train_param = []
        self.train_param.extend(list(self.model.parameters()))
        for i in range(self.agent_num):
            self.train_param.extend(self.model.agents[i].train_param)

        self.optimizer = torch.optim.Adam(self.train_param, lr=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.995)


    def weighted_reward(self, reward, alpha):
        reward_list = []
        for i in range(self.batch_size):
            if reward[i]==0:
                reward_list.append(-4*alpha[i])
            else:
                reward_list.append(4*alpha[i])
        
        return torch.stack(reward_list, dim=0)
    
    def reward_function(self, alpha_list):
        reward_batch = []
        reward_agent = []
        reward_list = []

        for t in alpha_list:
            for agent in t:
                for batch in agent:
                    reward_batch.append(entropy(batch.detach().cpu(), base=self.agent_num))
                reward_agent.append(torch.mean(torch.FloatTensor(reward_batch)))
                reward_batch=[]
            reward_list.append(torch.FloatTensor(reward_agent))
            reward_agent = []
        print(reward_list)
        
        print('reward list', len(reward_list))
        
        return(reward_list)

    def terminate_reward_function(self, T_reward, pred, label):
        correct = AvgMeter()
        acc_t = []
        correct_count = 0
        threshold = 0.8

        for p in pred:
            for i in range(len(p)):
                correct.update(p[i]==label[i])
            acc_t.append(correct.avg)

        for acc in acc_t:
            if acc >= threshold:
                correct_count += 1
            
        if acc_t[-1] >= threshold and correct_count == 1:
            terminate_reward = T_reward + 5
        if acc_t[-1] >= threshold and correct_count > 1:
            terminate_reward = T_reward + 3
        if acc_t[-1] < threshold:
            terminate_reward = T_reward -3
        if acc_t[-1] < threshold and correct_count > 0:
            terminate_reward = T_reward - 5
        return terminate_reward

    def train(self):
        if self.resume:
            self.load_ckpt(is_best=False)
        for epoch in range(self.start_epoch, self.epoch_num):
            #train_loss, train_rl, train_act, train_base, train_T, train_g_num, train_acc = self.train_one_epoch(epoch)
            train_acc, train_loss, train_g_num = self.train_one_epoch(epoch)
            #val_loss, val_rl, val_act, val_base, val_T, val_g_num, val_acc = self.validate(epoch)
            val_acc, val_loss, val_g_num = self.validate(epoch)
            #train_loss, train_rl, train_act, train_base, train_g_num, train_acc = self.train_one_epoch(epoch)
            #val_loss, val_rl, val_act, val_base, val_g_num, val_acc = self.validate(epoch)
            if val_acc > self.best_val_acc:
                is_best = True
                self.best_val_acc = val_acc
            self.scheduler.step()
            print("INFO - val_acc: {:.2f} - train_acc: {:.2f} - loss: {:.2f}".format(val_acc, train_acc, train_loss))
            writer.add_scalar('train acc', train_acc, epoch)
            writer.add_scalar('train glimpse number', train_g_num, epoch)
            writer.add_scalar('train loss', train_loss, epoch)
            writer.add_scalar('val acc', val_acc, epoch)
            writer.add_scalar('val glimpse number', val_g_num, epoch)
            writer.add_scalar('val loss', val_loss, epoch)
            writer.add_scalar('best val acc', self.best_val_acc, epoch)
            writer.flush()
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
        print('EPOCH:', epoch, end=' / ')
        print('best val_acc: {:.2f}'.format(self.best_val_acc), end=' / ')
        print('lr: %.6f' % self.optimizer.param_groups[0]['lr'])

        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        avg_g_num = AvgMeter()
        loss_rl, loss_act, loss_base, loss_T = 0, 0, 0, 0
        g_num_list = []
        start_t = time.time()
        with tqdm(self.train_loader, total=len(self.train_loader)) as pbar:
            for j, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                imgs = images.repeat(self.M, 1, 1, 1)
                label = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                
                log_probs, prob_ts, terminate_ts, T_reward_ts, l_list, b_list, log_pi_list, alpha_list, g_num = self.model(imgs)
                #l_list[time_step, agent_num, [batch_size, 2]]
                log_pi = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
                log_pi = log_pi.reshape((self.M, -1, log_pi.shape[-2], log_pi.shape[-1]))#[M, batch_size, time_step, agent_num]
                log_pi = torch.mean(log_pi, dim=0)#[batch_size, time_step, agent_num]
                baselines = torch.stack(b_list, dim=1)
                print('number of glimpse', g_num)
                print(baselines.shape)
                baselines = baselines.reshape((self.M, -1, baselines.shape[-2], baselines.shape[-1]))#[M, batch_size, time_step, agent_num]
                baselines = torch.mean(baselines, dim=0)
                #log_probs = torch.stack(log_prob_list, dim=1)
                log_probs = log_probs.view((self.M, -1, self.batch_size, log_probs.shape[-1]))
                log_probs = torch.mean(log_probs, dim=0) #[time_step, batch_size, num_of_class]
                predicted = torch.max(log_probs, 2)[1]  #indices store in element[1]
                correct = (predicted[-1] == label).float().detach()
                terminate_reward = self.terminate_reward_function(T_reward_ts, predicted.detach(), label)
                reward = self.reward_function(alpha_list)
                 
                #reward = reward_tensor.unsqueeze(1).repeat(1,g_num,1) #[batch_size, time_step, agent_num]
                adjusted_reward = reward - baselines.detach()
                
                loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)#sum along all glimpses
                
                loss_reinforce = torch.mean(loss_reinforce, dim=0).sum() / self.batch_size #actor
                #mean along all batch then sum up
                loss_baseline = F.mse_loss(baselines, reward) #critic
                loss_action = F.nll_loss(log_probs[-1], label) #classification
                prob_ts = torch.unsqueeze(prob_ts, 1)
                loss_terminate = F.binary_cross_entropy_with_logits(prob_ts, terminate_ts)
                loss_terminate = torch.mean(loss_terminate * terminate_reward)
                loss = loss_action + loss_reinforce + loss_baseline + loss_terminate
                
                avg_acc.update(correct.mean().item())
                avg_loss.update(loss.item())
                avg_g_num.update(g_num)

                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.train_param, max_norm=5.0)
                self.optimizer.step()
                if self.duration > self.save_gap and iteration == 0:
                	draw(images, l_list, label, predicted[-1], self.batch_size, self.agent_num, self.glimpse_size, g_num, self.patch_num, self.scale, self.M, epoch, 'train')
                end_t = time.time()
                self.epoch_time = end_t - start_t
        
                iteration = iteration + 1
        
        return 100*avg_acc.avg, avg_loss.avg, avg_g_num.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        iteration = 0
        acc_list, loss_list, reward_list = [], [], []
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        avg_g_num = AvgMeter()
        loss_rl, loss_act, loss_base, loss_T = [], [], [], []
        g_num_list = []
        
        for i, (images, labels) in enumerate(self.val_loader):
            images = images.to(self.device)
            imgs = images.repeat(self.M, 1, 1, 1)
            label = labels.to(self.device)
            
            #prob_ts, terminate_ts, T_reward_ts, l_list, b_list, log_pi_list, log_probs, alpha, g_num = self.model(imgs, self.train_terminate, sampling=False)
            log_probs, prob_ts, terminate_ts, T_reward_ts, l_list, b_list, log_pi_list, alpha, g_num = self.model(imgs)
            #l_list, b_list, log_pi_list, log_probs, alpha, g_num = self.model(imgs)

            log_pi = torch.stack(log_pi_list, dim=1) #[batch_size, time_step, agent_num]
            log_pi = log_pi.reshape((self.M, -1, log_pi.shape[-2], log_pi.shape[-1]))
            log_pi = torch.mean(log_pi, dim=0)
            baselines = torch.stack(b_list, dim=1) #[batch_size, time_step, agent_num]
            baselines = baselines.reshape((self.M, -1, baselines.shape[-2], baselines.shape[-1]))
            baselines = torch.mean(baselines, dim=0)
            #log_probs = torch.stack(log_prob_list, dim=1)
            log_probs = log_probs.view((self.M, -1, self.batch_size, log_probs.shape[-1]))
            log_probs = torch.mean(log_probs, dim=0) #[time_step, batch_size, num_of_class]
            predicted = torch.max(log_probs, 2)[1]  #indices store in element[1]
            correct = (predicted[-1] == label).float().detach()
            terminate_reward = self.terminate_reward_function(T_reward_ts, predicted.detach(), label)
            reward = self.weighted_reward(correct, alpha)
            
            
            reward = reward.unsqueeze(1).repeat(1,g_num,1)
            #reward[batch_size, time_step, agent_num]
            adjusted_reward = reward - baselines.detach()
            
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)#sum along all glimpses
            #loss_reinforce = torch.sum(-log_pi * 1, dim=1)#sum along all glimpses
            
            loss_reinforce = torch.mean(loss_reinforce, dim=0).sum() #actor
            #mean along all batch then sum up
            loss_baseline = F.mse_loss(baselines, reward) #critic
            loss_action = F.nll_loss(log_probs[-1], label) #classification
            prob_ts = torch.unsqueeze(prob_ts, 1)
            loss_terminate = F.binary_cross_entropy_with_logits(prob_ts, terminate_ts)
            loss_terminate = torch.mean(loss_terminate * terminate_reward)
            loss = loss_action + loss_reinforce + loss_baseline + loss_terminate
            
            avg_acc.update(correct.mean().item())
            avg_loss.update(loss.item())
            avg_g_num.update(g_num)
            
            if self.duration > self.save_gap and iteration == 0:
                draw(images, l_list, label, predicted[-1], self.batch_size, self.agent_num, self.glimpse_size, g_num, self.patch_num, self.scale, self.M, epoch, 'val', iteration)
            
            iteration = iteration + 1
   
        return 100*avg_acc.avg, avg_loss.avg, avg_g_num.avg
    
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
    trainer.train()






