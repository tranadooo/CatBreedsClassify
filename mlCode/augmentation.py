##数据增强#######
##必须要有base_path，下面必须要有基础数据的文件夹images，调整这两个变量值即可进行数据增强
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

def SaltAndPepper(image,percetage):  
    SP_NoiseImg=image.copy()
    SP_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
    for i in range(SP_NoiseNum): 
        randR=np.random.randint(0,image.shape[0]-1) 
        randG=np.random.randint(0,image.shape[1]-1) 
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0: 
            SP_NoiseImg[randR,randG,randB]=0 
        else: 
            SP_NoiseImg[randR,randG,randB]=255 
    return SP_NoiseImg 
def addGaussianNoise(image,percetage): 
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
    for i in range(G_NoiseNum): 
        temp_x = np.random.randint(0,h) 
        temp_y = np.random.randint(0,w) 
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0] 
    return G_Noiseimg
#dimming
def darker(image,percetage=0.9):
    image_darker= (image*percetage).astype(np.uint8)
    return image_darker

def brighter(image, percetage=1.5):
    image_brighter=np.clip((image*percetage),0,255).astype(np.uint8)
    return image_brighter

def rotate(image, angle=15, scale=1):
    rows, cols, channels = image.shape
    rotate = cv2.getRotationMatrix2D((rows*0.5, cols*0.5), angle, scale)
    '''
    第一个参数：旋转中心点
    第二个参数：旋转角度
    第三个参数：缩放比例
    '''
    image_rotate = cv2.warpAffine(image, rotate, (cols, rows))
    return image_rotate

def img_augmentation(filename, augment_times=1):   
    """
    args:
        filename: 需要增强的图片文件路径（相对路径，绝对路径都行）
        augment_times：需要增强的次数
    
    eff:将变换后的图片写入原文件同级目录下
    
    """
    
    img = cv2.imread(filename)
    if img is None:
        return 0
    
    method_list = ["SaltAndPepper","addGaussianNoise","brighter","darker","rotate"]
    
    for i in range(augment_times):
        
        method = random.choice(method_list)
        if method=="SaltAndPepper":
            img1=SaltAndPepper(img, 0.3)
        elif method=="addGaussianNoise":
            img1=addGaussianNoise(img, 0.3)
        elif method=="brighter":
            img1=brighter(img,percetage=np.random.uniform(1,2))
        elif method=="darker":
            img1=darker(img, np.random.uniform(0,1))
        elif method=="rotate":
            img1=rotate(img, angle=int(np.random.uniform(0,45)))
            
        cv2.imwrite(filename.split(".")[0]+"_%s"%(i+1)+".jpg", img1)
   
    
    
###test
# image_path='input/test/12136161_252.jpg' 
# li=img_augmentation(image_path,6)
