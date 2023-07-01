import numpy as np
import shutil
import os

AUG_NAMES = ['rotate', 'Crop', 'ZheDang', 'Zhedang']
def is_aug(it_name):
    for aug_name in AUG_NAMES:
        if aug_name in it_name: return True
    return False

def main():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    SrcDir = os.path.join(working_dir, 'label_aug')
    DstDir = os.path.join(working_dir, 'label')
    
    if not os.path.exists(DstDir): os.makedirs(DstDir)
    data_list = os.listdir(SrcDir)
    for it in data_list:
        if not (it[-4:]=='.png' and is_aug(it)):
            src_path = os.path.join(SrcDir, it)
            dst_path = os.path.join(DstDir, it)
            shutil.copyfile(src_path, dst_path)
            print(it)

if __name__ == '__main__':
    main()