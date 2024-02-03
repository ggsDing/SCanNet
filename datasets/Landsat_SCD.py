import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F

num_classes = 5
ClASSES = ['0: No change', '1: Farmland', '2: Desert', '3: Building', '4: Water']
COLORMAP = [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]]

#MEAN_A = np.array([92.32, 88.98, 86.87])
#STD_A  = np.array([40.41, 40.18, 39.25])
#MEAN_B = np.array([86.34, 85.10, 83.15])
#STD_B  = np.array([44.63, 43.29, 43.26])

MEAN_A = np.array([141.53, 139.20, 137.73] )
STD_A  = np.array([81.99, 83.31, 83.89])
MEAN_B = np.array([137.36, 136.50, 135.144])
STD_B  = np.array([85.97, 86.01, 86.81])


root = 'D:/RS/Landsat/'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = np.asarray(ColorLabel, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap.astype(np.int8)

def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def tensor2int(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = im * STD_A + MEAN_A
    else:
        im = im * STD_B + MEAN_B
    return im.astype(np.uint8)

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs

def read_RSimages(mode, rescale=False):
    assert mode in ['train', 'val', 'test']
    list_path=os.path.join(root, mode+'_list.txt')
    img_A_dir = os.path.join(root, 'A')
    img_B_dir = os.path.join(root, 'B')
    label_A_dir = os.path.join(root, 'labelA')
    label_B_dir = os.path.join(root, 'labelB')
    
    list_info = open(list_path, 'r')
    data_list = list_info.readlines()
    data_list = [item.rstrip() for item in data_list]
    
    imgsA_list, imgsB_list, labelsA, labelsB = [], [], [], []
    count = 0
    for it in data_list:
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            
            imgsA_list.append(img_A_path)
            imgsB_list.append(img_B_path)
            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)
            labelsA.append(label_A)
            labelsB.append(label_B)
        count+=1
        if not count%100: print('%d/%d images loaded.'%(count, len(data_list)))
        #if count>99: break
    print(str(len(imgsA_list)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgsA_list, imgsB_list, labelsA, labelsB

class Data(data.Dataset):
    def __init__(self, mode, random_flip=False, valid_label=False):
        self.random_flip = random_flip
        self.valid_label = valid_label
        self.imgsA_list, self.imgsB_list, self.labelsA, self.labelsB = read_RSimages(mode)
        self.len = len(self.imgsA_list)
    
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgsA_list[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgsA_list[idx])
        img_B = io.imread(self.imgsB_list[idx])
        if self.valid_label:
            invalid_labelA = np.all(img_A == 255, axis=2)
            invalid_labelB = np.all(img_B == 255, axis=2)
            invalid_label = invalid_labelA * invalid_labelB
            valid_label = np.logical_not(invalid_label)
        label_A = self.labelsA[idx]
        label_B = self.labelsB[idx]
        #label_A = Color2Index(label_A)
        #label_B = Color2Index(label_B)
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        if self.valid_label:
            if self.random_flip:
                img_A, img_B, label_A, label_B, valid_label = transform.rand_rot90_flip_SCD5([img_A, img_B, label_A, label_B, valid_label])
            return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B), torch.from_numpy(valid_label)
        else:
            if self.random_flip:
                img_A, img_B, label_A, label_B = transform.rand_rot90_flip_SCD([img_A, img_B, label_A, label_B])
            return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)

    def __len__(self):
        return len(self.imgsA_list)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
                
        imgA_dir = os.path.join(test_dir, 'A')
        imgB_dir = os.path.join(test_dir, 'B')
        list_path=os.path.join(test_dir, 'test_list.txt')
        list_info = open(list_path, 'r')
        data_list = list_info.readlines()
        data_list = [item.rstrip() for item in data_list]
        
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len
