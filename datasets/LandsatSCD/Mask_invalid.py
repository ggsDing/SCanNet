import numpy as np
from skimage import io
import os

def main():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    ImgDirA = os.path.join(working_dir, 'A')
    ImgDirB = os.path.join(working_dir, 'B')
    LabelDirA = os.path.join(working_dir, 'labelA')
    LabelDirB = os.path.join(working_dir, 'labelB')
    LabelDirA_rgb = os.path.join(working_dir, 'labelA_rgb')
    LabelDirB_rgb = os.path.join(working_dir, 'labelB_rgb')
        
    data_list = os.listdir(ImgDirA)
    for idx, it in enumerate(data_list):
        if (it[-4:]=='.png'):
            Img_pathA = os.path.join(ImgDirA, it)
            Img_pathB = os.path.join(ImgDirB, it)
            label_pathA = os.path.join(LabelDirA, it)
            label_pathB = os.path.join(LabelDirB, it)
            label_pathA_rgb = os.path.join(LabelDirA_rgb, it)
            label_pathB_rgb = os.path.join(LabelDirB_rgb, it)
            
            imgA = io.imread(Img_pathA)
            imgB = io.imread(Img_pathB)
            labelA = io.imread(label_pathA)
            labelB = io.imread(label_pathB)
            labelA_rgb = io.imread(label_pathA_rgb)
            labelB_rgb = io.imread(label_pathB_rgb)
            invalid_maskA = np.all(imgA==255, axis=2)
            invalid_maskB = np.all(imgB==255, axis=2)
            invalid_mask = invalid_maskA*invalid_maskB
            #labelA = labelA-invalid_mask.astype(np.uint8)
            #labelB = labelB-invalid_mask.astype(np.uint8)
            valid_mask = np.logical_not(invalid_mask)
            valid_mask = np.expand_dims(valid_mask,2).repeat(3,axis=2)
            labelA_rgb = labelA_rgb*valid_mask
            labelB_rgb = labelB_rgb*valid_mask
            
            #io.imsave(label_pathA, labelA, check_contrast=False)
            #io.imsave(label_pathB, labelB, check_contrast=False)
            io.imsave(label_pathA_rgb, labelA_rgb, check_contrast=False)
            io.imsave(label_pathB_rgb, labelB_rgb, check_contrast=False)
            
            if not idx%50: print('%d/%d labels processed.'%(idx, len(data_list)))
    print('Done.')


if __name__ == '__main__':
    main()
