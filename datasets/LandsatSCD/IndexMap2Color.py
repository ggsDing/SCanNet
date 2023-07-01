import numpy as np
from skimage import io
import os

num_classes = 10
COLORMAP = [[255, 255, 255], [255, 165, 0], [230, 30, 100], [70, 140, 0], [218, 112, 214], [0, 170, 240], [127, 235, 170], [230, 80, 0], [205, 220, 57], [218, 165, 32]]
CLASSES = ['No change', 'Farmland-desert', 'Farmland-building', 'Desert-farmland', 'Desert-building', 'Desert-water', 'Building-farmland', 'Building-desert', 'water-farmland', 'water-desert']

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Index2Color(mask):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(mask, dtype='int32')
    return colormap[x, :]
    
def Index2Color_bn(mask):
    mask = np.logical_not(mask) 
    mask = np.asarray(mask, dtype='uint8')
    return mask*255
    
def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)

def main():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    SrcDir = os.path.join(working_dir, 'label')
    DstDir = os.path.join(working_dir, 'label_rgb')
    
    if not os.path.exists(DstDir): os.makedirs(DstDir)
    data_list = os.listdir(SrcDir)
    for it in data_list:
        if (it[-4:]=='.png'):
            img_src_path = os.path.join(SrcDir, it)
            img_dst_path = os.path.join(DstDir, it)
            label = io.imread(img_src_path)
            #label = Color2Index(label)
            label_new = Index2Color(label)
            io.imsave(img_dst_path, label_new)
            print('Process finished: '+img_dst_path)

if __name__ == '__main__':
    main()