# SCanNet
Pytorch codes of 'Joint Spatio-Temporal Modeling for Semantic Change Detection in Remote Sensing Imagess' [[paper]](http://xxx)


![alt text](https://github.com/ggsDing/WiCoNet/blob/main/WiCoNet.png)
![alt text](https://github.com/ggsDing/WiCoNet/blob/main/data_BLU.png)

**To be updated:**
- [x] Codes for the BLU dataset
- [x] Codes for the GID
- [x] Codes for the Potsdam dataset
- [ ] Optimizing the codes to easily switch datasets

**How to Use**
1. Split the data into training, validation and test set and organize them as follows:

>YOUR_DATA_DIR
>  - Train
>    - image
>    - label
>  - Val
>    - image
>    - label
>  - Test
>    - image
>    - label

2. Change the training parameters in *Train_WiCo_BLU.py*, especially the data directory.

3. To evaluate, change also the parameters in *Eval_WiCo_BLU.py*, especially the data directory and the checkpoint path.

