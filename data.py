# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from numpy import random
from skimage import io
from skimage import draw
import math
import pickle

#極大値を取得、in_imgs = ndarray[N][0][imgH][imgW]
def get_local_max_point(in_imgs, threshold = 0.5):
    
    m_imgs = chainer.Variable(in_imgs)
    m_imgs = F.max_pooling_2d(m_imgs, ksize=9 ,stride=1, pad=4)
    m_imgs = m_imgs.data
    p_array = (in_imgs == m_imgs)#極大値判定（True or False）の配列
    out_imgs = in_imgs * p_array
    
    out_imgs[out_imgs >= threshold]=1
    out_imgs[out_imgs < threshold]=0
    
    return out_imgs

#データの切り出し        
def get_data_N_rand(DataO, N_pic =1, imgH = 256, imgW = 256):
  
    Data={}
    Data['N_pic'] = N_pic
    Data['imgH'] = imgH
    Data['imgW'] = imgW    
    
    #切り出したデータの保存先 dim=[N][0][imgH][imgW] ,float32       
    for key in ['x', 't_point', 't_core']:
        Data[key] = np.zeros((N_pic, 1, imgH, imgW),dtype= np.float32)

    #切り出し限界を設定
    xlim = DataO['imgW'] - imgW + 1
    ylim = DataO['imgH'] - imgH + 1     

    for i in range(0, N_pic):            
        im_num =np.random.randint(0, DataO['N_pic'])#切り取る写真の番号
        rotNo = np.random.randint(8) #回転No
        cutx = np.random.randint(0, xlim)
        cuty = np.random.randint(0, ylim)

        #切り取りと回転
        for key in ['x', 't_point', 't_core']:
            Data[key][i][0] = rand_rot((DataO[key][im_num][0][cuty: cuty+imgH, cutx: cutx+imgW]), rotNo)
    
    return Data   
                
#np配列をもらって左右上下の反転・90、180、270°の回転した配列を返す
def rand_rot(a,rotNo):    
    #i = np.random.randint(8)
    i=rotNo
    if i==0:
        b=a
    elif i==1:
        b=np.fliplr(a)
    elif i==2:
        b=np.flipud(a)        
    elif i==3:
        b=np.rot90(a,1)     
    elif i==4:
        b=np.rot90(a,2)        
    elif i==5:
        b=np.rot90(a,3)
    elif i==6:
        b=np.fliplr(np.rot90(a,1))
    elif i==7:
        b=np.flipud(np.rot90(a,1))         
    return b 


#circleを描画 in_imgs = ndarray[N][0][imgH][imgW]
def draw_circle(in_imgs):
        
    cir = np.zeros((1,1,15,15), dtype= np.float32)
    rr, cc = draw.circle_perimeter(7,7,5)
    cir[0][0][rr, cc] = 1      
    decon_cir = L.Deconvolution2D(1, 1, 15, stride=1, pad=7)
    decon_cir.W.data = cir
    out_imgs  = decon_cir(chainer.Variable(in_imgs))
    out_imgs = out_imgs.data
    return out_imgs

#コアを描画、in_imgs = ndarray[N][0][imgH][imgW]
def draw_core(in_imgs, max_xy = 15, sig=3.0):
    
    sig2=sig*sig
    c_xy=7
    core=np.zeros((max_xy, max_xy),dtype = np.float32)
    for px in range(0, max_xy):
        for py in range(0, max_xy):
            r2 = float((px-c_xy)*(px-c_xy)+(py-c_xy)*(py-c_xy))
            core[py][px] = math.exp(-r2/sig2)*1
    core = core.reshape((1,1,core.shape[0],core.shape[1]))  
    
    decon_core = L.Deconvolution2D(1, 1, max_xy, stride=1, pad=7)
    decon_core.W.data = core
    out_imgs  = decon_core(chainer.Variable(in_imgs))
    out_imgs = out_imgs.data
    return out_imgs

#xとtのデータを取得
def get_ori_data_pos_1pic(N_pic = 1, imgH = 512, imgW = 512 ,fpath = 'imgA.png'):
    
    Data={}    
    Data['N_pic'] = N_pic
    Data['imgH'] = imgH
    Data['imgW'] = imgW
    
    Data['x'] = np.zeros((N_pic, 1, imgH, imgW), dtype= np.float32)    
    Data['t_point'] = np.zeros((N_pic, 1, imgH, imgW), dtype= np.float32)

    img = io.imread(fpath)
    img = img[0:imgH, 0:imgW]
  
    Data['x'][0][0] = img.astype(np.float32) / 65535

    with open("posA.pkl", "rb") as pk_data:
        pos = pickle.load(pk_data)    
    
    for p in pos:
        Data['t_point'][0][0][p[0]][p[1]] =1
     
    Data['t_core'] = draw_core(Data['t_point'])
    
    return Data

#xのデータ取得
def get_ori_data_x_1pic(N_pic = 1, imgH = 512, imgW = 512 ,fpath = 'imgB.png'):
    
    Data={}    
    Data['N_pic'] = N_pic
    Data['imgH'] = imgH
    Data['imgW'] = imgW
    
    Data['x'] = np.zeros((N_pic, 1, imgH, imgW), dtype= np.float32) 
    img = io.imread(fpath)
    img = img[0:imgH, 0:imgW]  
    Data['x'][0][0] = img.astype(np.float32) / 65535
       
    return Data