"""
utils class function
"""
import os 
import cv2
import random
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical 
from config import UnetConfig
import keras as K
#设置全局变量
config = UnetConfig()
img_w = config.IMAGE_W
img_h = config.IMAGE_H
n_label = config.CLASSES_NUM

class DataGenerator_wqw(K.utils.Sequence):
    def __init__(self,img_path,label_path,width,height,batch_size,num_class,config):
        self.config=config
        self.num_class=num_class
        self.width=width
        self.height=height
        self.batch_size=batch_size
        imgdir=os.listdir(img_path)
        labeldir=os.listdir(label_path)
        assert len(imgdir)==len(labeldir),"the count of img and label is not equality"
        self.imgdir=[]
        self.labeldir=[]
        for l in imgdir:
            self.imgdir.append(img_path+"/"+l)
        for l in imgdir:
            self.labeldir.append(label_path+"/"+l)
        self.image_id=np.arange(len(self.imgdir))

    def __len__(self):
        return int(np.ceil(len(self.imgdir)/float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.image_id)

    def __getitem__(self,idx):
        id=self.image_id[idx*self.batch_size:(idx+1)*self.batch_size]
        images=[]
        labels=[]
        for i in id:
            img=cv2.imread(self.imgdir[i],-1)
            img=cv2.resize(img,(self.config.IMAGE_H,self.config.IMAGE_W))
            label=cv2.imread(self.labeldir[i])
            label=cv2.resize(label,(self.config.IMAGE_H,self.config.IMAGE_W),cv2.INTER_NEAREST)
            label=label[:,:,0]

            label=(label==1)
            label=np.array(label,dtype=int)

            img=np.asarray(img,'f')
            img=img-self.config.MEAN
            img=img/255
            label=np.asarray(label,'f')
            images.append(img)
            labels.append(label)
        images=np.array(images)
        labels=to_categorical(labels,self.num_class)
        #labels=labels.reshape(labels.shape[0],-1,labels.shape[-1])
        return images,labels


