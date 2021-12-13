import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import HRSC2016
from tqdm import tqdm


from torch.utils.data.dataset import Dataloader, Dataset

def train_model():
    epochs = 100
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('./weight'):
        os.mkdir('./weight')
    
    ds = HRSC2016()
    loader = Dataloader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True)

    model = RecurrentAttention()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.7, 0.9]], gamma=0.1)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(0, epochs):
        pbar = tqdm(enumerate(loader), total=len(loader))
        mloss = torch.zeros(2).cuda()

        for i, (ni, batch) in enumerate(pbar):
            model.train()
            optimizer.zero_grad()
            imgs, bboxes = batch['image'], batch['boxes']
            if torch.cuda.is_available():
                imgs, bboxes = imgs.cuda(), bboxes.cuda()
            losses = model(imgs, bboxes)
