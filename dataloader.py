import torchvision
import torch
import cv2 as cv
import xml.etree.ElementTree as ET

from torch.utils.data.dataset import DataLoader, Dataset

def xmlParser(fileName):
    try:
        root = ET.parse(fileName)
        object_num = 0
        for obj in root.iter('HRSC_Object'):
            object_num = object_num + 1
        return object_num
    except:
        pass
def dataPreporcess():
    object_num_list = []
    for i in range(1,1681):
        name = './dataset/labels/1' + str(i).zfill(8) + '.xml'
        object_num = xmlParser(name)
        if object_num != None:
            object_num_list.append(object_num)

    print(object_num_list)

class HRSCDataloader(Dataset):
    def __init__(self, rooot):
        pass
    def __getitem__(self):
        pass
    def __len__(self):
        pass



