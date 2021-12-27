import os
import torch
import cv2 as cv
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from bbox import mask_valid_boxes, constraint_theta


class HRSC2016(Dataset):
    
    def __init__(self, level=1):
        self.image_names_path = './HRSC2016/AllImages/HRSC_image_names.txt'
        self.image_list = self._load_image_names()
        self.level = level
        if self.level == 1:
            self.classes = ('__backround__', 'ship')
        self.class_num = len(self.classes)
        self.class_to_index = dict(zip(self.classes, range(self.class_num)))
        self.augment = False
    
    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = cv.cvtColor(cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        roidb = self._load_annotation(self.image_list[index])
        get_index = np.where(roidb['gt_classes'] != 0)[0]
        nt = len(roidb['boxes'])
        get_boxes = np.zeros((len(get_index), 6), dtype=np.float32)
        if nt:
            bboxes = roidb['boxes'][get_index, :]
            classes = roidb['gt_classes'][get_index]
            get_boxes[:, :-1] = bboxes

            mask = mask_valid_boxes(bboxes)
            bboxes = bboxes[mask]
            get_boxes = get_boxes[mask]
            classes = self.classes[mask]

            for i, bbox in enumerate(bboxes):
                get_boxes[i, 5] = classes[i]
            get_boxes = constraint_theta(get_boxes)
            cx, cy, w, h = [get_boxes[:, x] for x in range(4)]
            x1 = cx - 0.5*w
            x2 = cx + 0.5*w
            y1 = cy - 0.5*h
            y2 = cy + 0.5*h
            get_boxes[:, 0] = x1
            get_boxes[:, 1] = y1
            get_boxes[:, 2] = x2
            get_boxes[:, 3] = y2
        
        return {'image': img, 'boxes': get_boxes, 'path': img_path}
    
    def __len__(self):
        return len(self.image_list)
    
    def _load_image_names(self):
        assert os.path.exists(self.image_names_path), \
            'Path not exist: {}'.format(self.image_names_path)
        with open(self.image_names_path) as f:
            image_name_list = [i.strip() for i in f.readlines()]
        return image_name_list
    
    def _load_annotation(self, index):
        boxes, gt_classes = [], []
        root, name = os.path.split(index)
        xmlName = os.path.join(root.replace('AllImages', 'Annotations'), name[:-4]+'.xml')
        with open(xmlName, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            assert '<HRSC_Object>' in content, 'Background picture occured in %s' %xmlName
            objects = content.split('<HRSC_Object>')
            info = objects.pop(0)
            for obj in objects:
                assert len(obj) != 0, 'No object found in %s' %xmlName
                cls_id = obj[obj.find('<Class_ID>')+10 : obj.find('</Class_ID>')]
                diffculty = obj[obj.find('<difficult>')+11 : obj.find('</difficult>')]
                if diffculty == '1':
                    continue
                cx = round(eval(obj[obj.find('<mbox_cx>')+9 : obj.find('</mbox_cx>')]))
                cy = round(eval(obj[obj.find('<mbox_cy>')+9 : obj.find('</mbox_cy>')]))
                w = round(eval(obj[obj.find('<mbox_w>')+8 : obj.find('</mbox_w>')]))
                h = round(eval(obj[obj.find('<mbox_h>')+8 : obj.find('</mbox_h>')]))
                a = eval(obj[obj.find('<mbox_ang>')+10 : obj.find('</mbox_ang>')])/math.pi*180
                box = np.array([cx, cy, w, h, a])
                boxes.append(box)
                label = self.class_mapping(cls_id, self.level)
                gt_classes.append(label)
        return {'boxes': np.array(boxes), 'gt_classes': np.array(gt_classes)}
    
    def class_mapping(self, cls_id, level):
        if level == 1:
            return 1


