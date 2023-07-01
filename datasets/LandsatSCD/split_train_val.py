import numpy as np
from skimage import io
import random
import os

def main():    
    val_ratio = 1/5.0
    test_ratio = 1/5.0
    root_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(root_dir, 'A')
    train_info = open(os.path.join(root_dir, 'train_list.txt'), 'w')
    val_info = open(os.path.join(root_dir, 'val_list.txt'), 'w')
    test_info = open(os.path.join(root_dir, 'test_list.txt'), 'w')
        
    img_list = os.listdir(img_dir)
    valid_list = []
    train_list = []
    val_list = []
    test_list = []
    
    for it in img_list:
        it_ext=it[-4:]
        if (it_ext=='.png'): valid_list.append(it)
    num_pics = len(valid_list)
    num_val = int(num_pics*val_ratio)
    num_test = int(num_pics*test_ratio)
    num_train = num_pics - num_val - num_test
    print('num_pics: %d, num_train: %d, num_val:%d, num_test:%d.'%(num_pics, num_train, num_val, num_test))
    
    random.shuffle(valid_list)
    for idx, it in enumerate(valid_list):
        if idx<num_train: train_list.append(it)
        elif idx<(num_train+num_val): val_list.append(it)
        else: test_list.append(it)
    for it in train_list: train_info.write(it+'\n')
    for it in val_list: val_info.write(it+'\n')
    for it in test_list: test_info.write(it+'\n')
    

if __name__ == '__main__':
    main()