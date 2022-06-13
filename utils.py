import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

def rescale(im, target_size, max_size, keep_ratio, multiple=32):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if keep_ratio:
        # method1
        im_scale = float(target_size) / float(im_size_min)  
        if np.round(im_scale * im_size_max) > max_size:     
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
        # method2
        # im_scale = float(target_size) / float(im_size_max)
        # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # im_scale = np.array([im_scale, im_scale, im_scale, im_scale])

    else:
        target_size = int(np.floor(float(target_size) / multiple) * multiple)
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    return im, im_scale


class Rescale(object):
    def __init__(self, target_size=600, max_size=2000, keep_ratio=True):
        self._target_size = target_size
        self._max_size = max_size
        self._keep_ratio = keep_ratio

    def __call__(self, im):
        if isinstance(self._target_size, list):
            random_scale_inds = npr.randint(0, high=len(self._target_size))
            target_size = self._target_size[random_scale_inds]
        else:
            target_size = self._target_size
        im, im_scales = rescale(im, target_size, self._max_size, self._keep_ratio)
        return im, im_scales


class Normailize(object):
    def __init__(self):
        # RGB: https://github.com/pytorch/vision/issues/223
        self._transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, im):
        im = self._transform(im)
        return im


class Reshape(object):
    def __init__(self, unsqueeze=True):
        self._unsqueeze = unsqueeze
        return

    def __call__(self, ims):
        if not torch.is_tensor(ims):
            ims = torch.from_numpy(ims.transpose((2, 0, 1)))
        if self._unsqueeze:
            ims = ims.unsqueeze(0)
        return ims

def denormalize(input_tensor):
    inverse = transforms.Compose([
        transforms.Normalize((0., 0., 0.), (1/0.229, 1/0.224, 1/0.225)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.))
    ])
    return inverse(input_tensor)

def draw_text(src, exist, pred):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'class' + str(exist)
    text2 = 'pred:' + str(pred)
    cv2.putText(src, text1, (20,20), font, 0.8, (0,0,255), 2)
    cv2.putText(src, text2, (20,45), font, 0.8, (0,0,255), 2)

def draw_glimpse(src, x, y, k, s, size):
    for i in range(k):
        start_x = round(x-size/2)
        start_y = round(y-size/2)
        end_x = round(x+size/2)
        end_y = round(y+size/2)
        cv2.rectangle(src, (start_x, start_y), (end_x, end_y), (0,0,255), 1)
        size = size * s
    
    

def draw(imgs, l_list, exist, pred, batch_size, agent_num, g_size, g_num, k, s, m, epoch, name, batch=0):
    #imgs: [batch_size,channel,width,height]
    #l_list: [glimpse_num, agent_num, [batch_size, location]]
    imgs = imgs.cpu()
    exist = exist.cpu().numpy()
    pred = pred.cpu().detach().numpy()
    save_path = os.path.join('./results', name, str(epoch))
    if batch == 0:
        os.makedirs(save_path)

    for i in range(len(imgs)):
        array = denormalize(imgs[i]).numpy()
        maxval = array.max()
        array = array*255/maxval
        src = np.uint8(array)
        src = src.transpose(1,2,0)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        mat_list = []
        for j in range(g_num):
            mat = src.copy()
            for l in range(agent_num):
                x = round((l_list[j][l][i*m-1][0].item() + 1) * 32 / 2)
                y = round((l_list[j][l][i*m-1][1].item() + 1) * 32 / 2)
                draw_glimpse(mat, x, y, k, s, g_size)
            mat_list.append(mat)
        output = mat_list[0]
        for j in range(len(mat_list)-1):
            output = np.hstack((output,mat_list[j+1]))
        output = cv2.resize(output, (300*g_num, 300))
        draw_text(output, exist[i], pred[i])
        #cv2.imshow('result', output)
        #cv2.waitKey(1)
        if batch == 0:
            cv2.imwrite(save_path + '/' + str(i) + '.jpg', output)
        else:
            cv2.imwrite(save_path + '/' + str(batch) + str(i) + '.jpg', output)
                
