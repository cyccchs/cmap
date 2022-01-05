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
    batch_size=4,
    num_workers=8,
    collate_fn=collater,
    shuffle=False)
pbar = tqdm(enumerate(loader), total=len(loader))
model = RecurrentAttention(4, 128, 128, 3, 3, 256, 4)
for i, (ni,batch) in enumerate(pbar):
    imgs, existence = batch['image'], batch['existence']
    b,c,h,w = imgs.shape
    g_t = model(imgs, existence)
