import numpy as np
from skimage import io
import os

MCD_numcls = 10
MCD_COLORMAP = [[255, 255, 255], [255, 165, 0], [230, 30, 100], [70, 140, 0], [218, 112, 214], [0, 170, 240], [127, 235, 170], [230, 80, 0], [205, 220, 57], [218, 165, 32]]
MCD_CLASSES = ['0: No change', '1: Farmland-desert', '2: Farmland-building', '3: Desert-farmland', '4: Desert-building', '5: Desert-water', '6: Building-farmland', '7: Building-desert',' 8: water-farmland', '9: water-desert']

SCD_ClASSES = ['0: No change', '1: Farmland', '2: desert', '3: building', '4: water']
SCD_COLORMAP = [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]]
MAP_A = [0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
MAP_B = [0, 2, 3, 1, 3, 4, 1, 2, 1, 2]

def is_img(ext):
    ext = ext.lower()
    if ext == '.jpg':
        return True
    elif ext == '.png':
        return True
    elif ext == '.jpeg':
        return True
    elif ext == '.bmp':
        return True
    elif ext == '.tif':
        return True
    else:
        return False

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(SCD_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        data = data.astype(np.int32)
        idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
        IndexLabels.append(colormap2label[idx])
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
    return colormap2label[idx]

def Index2Color(mask):
    colormap = np.asarray(SCD_COLORMAP, dtype='uint8')
    x = np.asarray(mask, dtype='int32')
    return colormap[x, :]

def mapping(MCD_idx):
    funcA = mappingA[MCD_idx]
    SCD_idxB = mappingB[MCD_idx]
    return SCD_idxA, SCD_idxB

def MCD2SCD(label):
    #h, w = Img.shape
    #funcA = lambda idx: mappingA[idx]
    #funcB = lambda idx: mappingB[idx]
    #labelA = funcA(label)
    #labelB = funcB(label)
    label = np.asarray(label)
    mapA = np.asarray(MAP_A)
    mapB = np.asarray(MAP_B)
    labelA = mapA[label]
    labelB = mapB[label]
    return labelA.astype(np.uint8), labelB.astype(np.uint8)

def main():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    SrcDir = os.path.join(working_dir, 'label')
    DstDirA = os.path.join(working_dir, 'labelA_rgb')
    DstDirB = os.path.join(working_dir, 'labelB_rgb')
    if not os.path.exists(DstDirA): os.makedirs(DstDirA)
    if not os.path.exists(DstDirB): os.makedirs(DstDirB)
    
    data_list = os.listdir(SrcDir)
    for idx, it in enumerate(data_list):
        if (it[-4:]=='.png'):
            src_path = os.path.join(SrcDir, it)
            dst_pathA = os.path.join(DstDirA, it)
            dst_pathB = os.path.join(DstDirB, it)
            label = io.imread(src_path)
            labelA, labelB = MCD2SCD(label)
            labelA = Index2Color(labelA)
            labelB = Index2Color(labelB)
            io.imsave(dst_pathA, labelA, check_contrast=False)
            io.imsave(dst_pathB, labelB, check_contrast=False)
            if not idx%50: print('%d/%d labels processed.'%(idx, len(data_list)))
    print('Done.')

if __name__ == '__main__':
    main()
