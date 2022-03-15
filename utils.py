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
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 均值和方差
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
    inverse = transforms.Compose([transforms.Normalize((0.,0.,0.),
                                                        (1/0.229, 1/0.224, 1/0.225)),
                                transforms.Normalize((-0.485,-0.456,-0.406),
                                                    (1.,1.,1.))])
    return inverse(input_tensor)

def draw_bbox(img, x, y, size, exist, pred, reward, i):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, 'class'+str(exist[0]), (20,20), font, 0.8, (0,0,255), 2)
    img = cv2.putText(img, 'pred: '+str(pred[0].item()), (20,40), font, 0.8, (0,0,255), 2)
    img = cv2.putText(img, 'reward: '+str(reward[0][0].item()), (20,60), font, 0.8, (0,0,255), 2)
    if i == 0:
        return cv2.rectangle(img, (round(x-size/2),round(y-size/2)), (round(x+size/2), round(y+size/2)), (255,0,0), 2)
    if i == 1:
        return cv2.rectangle(img, (round(x-size/2),round(y-size/2)), (round(x+size/2), round(y+size/2)), (0,255,0), 2)

def draw(imgs, l_list, existence, predicted, reward, epoch, size):
    #imgs: [batch_size,channel,width,height]
    #l_list: [glimpse_size, agent_num, [batch_size, location]]
    imgs = imgs.cpu()
    existence = existence.cpu().numpy()
    predictde = predicted.cpu()
    reward = reward.cpu()

    array = denormalize(imgs[0]).numpy()
    maxval = array.max()
    array = array*255/maxval
    mat = np.uint8(array)
    mat = mat.transpose(1,2,0)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    mat_list = None
    mat_list = []
    for i in range(2):
        mat_list.append(mat.copy())
    for j in range(8):
        x = round((l_list[0][j][0][0].item()/2+0.5)*800)
        y = round((l_list[0][j][0][1].item()/2+0.5)*800)
        draw_bbox(mat_list[0], x, y, size, existence, predicted, reward, 0)
    for j in range(8):
        x = round((l_list[len(l_list)-1][j][0][0].item()/2+0.5)*800)
        y = round((l_list[len(l_list)-1][j][0][1].item()/2+0.5)*800)
        draw_bbox(mat_list[1], x, y, size, existence, predicted, reward, 1)

    output = mat_list[0]
    for i in range(len(mat_list)-1):
        output = np.hstack((output, mat_list[i+1]))
    
    cv2.imshow('img', output)
    cv2.waitKey(1)
    cv2.imwrite('./results/'+str(epoch)+'.jpg', output)
