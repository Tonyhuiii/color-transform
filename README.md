# color-transform
origianl paper: A Style-Aware Content Loss for Real-time HD Style Transfer (https://arxiv.org/pdf/1807.10201.pdf).  
![](https://github.com/Tonyhuiii/color-transform/blob/master/1.jpeg).
## Environment
● Ubuntu 18.04  
● NVIDIA TITIAN RTX  
● CUDA 10.0 + CuDNN 
## Prerequisites
● Python 3.6  
● [PyTorch](https://pytorch.org/) 1.2.0 
### visdom
`pip install visdom`

## Prepare Dataset
● Build your own dataset by setting up the following directory structure:  
`datasets/<dataset_name>/train/A　　　# Contains domain A images (MUSE image)`  
`datasets/<dataset_name>/train/B　　　# Contains domain B images (HE image)`    　　　
`datasets/<dataset_name>/test/A　　　　# Contains domain A images (MUSE image)`  
`datasets/<dataset_name>/test/B　　　　# Contains domain A images (HE image)`  　　
## Train
● To view training results and loss plots, run `visdom` in another terminal and click the URL (http://localhost:8097).  
● Train the model:     
`python train.py --dataroot ./datasets/<dataset_name>/ --cuda`  
● checkpoint will be saved in folder:  
`checkpoint/<dataset_name>`

## Test
● Test the model:   
`python test.py --dataroot ./datasets/<dataset_name>/ --cuda`  
● result will be saved in folder:    
`output/<dataset_name>/fake_A`  
`output/<dataset_name>/fake_B`

## Acknowledgments 
Code is based on Pytorch implementation of CycleGAN (https://github.com/aitorzip/PyTorch-CycleGAN);  
Tensorflow implementation adaptive-style-transfer (https://github.com/CompVis/adaptive-style-transfer).  
