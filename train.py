import os
import argparse
from dataloader import HRSC2016
from tqdm import tqdm


from torch.utils.data.dataset import Dataloader, Dataset

def train_model():
    epochs = 5
    batch_size = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('./weight'):
        os.mkdir('./weight')
    
    ds = HRSC2016()
    loader = Dataloader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True)
    pbar = tqdm(enumerate(loader), total=len(loader))

