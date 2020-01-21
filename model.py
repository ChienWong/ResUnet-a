import keras.models as KM
import keras.layers as KL
import keras.engine as KE
import keras.backend as KB
from keras.utils import plot_model
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50

from config import UnetConfig
import utils
import math
import numpy as np
class UNet(object):
    def __init__(self, config=UnetConfig()):
        self.config = config
        self.model = self.build_model_resUnet()

    def build_model_resUnet(self):
        def Tanimoto_loss(label,pred):
            square=tf.square(pred)
            sum_square=tf.reduce_sum(square,axis=-1)
            product=tf.multiply(pred,label)
            sum_product=tf.reduce_sum(product,axis=-1)
            denomintor=tf.subtract(tf.add(sum_square,1),sum_product)
            loss=tf.divide(sum_product,denomintor)
            loss=tf.reduce_mean(loss)
            return 1.0-loss

        def Tanimoto_dual_loss(label,pred):
            loss1=Tanimoto_loss(pred,label)
            pred=tf.subtract(1.0,pred)
            label=tf.subtract(1.0,label)
            loss2=Tanimoto_loss(label,pred)
            loss=(loss1+loss2)/2

        def ResBlock(input,filter,kernel_size,dilation_rates,stride):
            def branch(dilation_rate):
                x=KL.BatchNormalization()(input)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                x=KL.BatchNormalization()(x)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                return x
            out=[]
            for d in dilation_rates:
                out.append(branch(d))
            if len(dilation_rates)>1:
                out=KL.Add()(out)
            else:
                out=out[0]
            return out
        def PSPPooling(input,filter):
            x1=KL.MaxPooling2D(pool_size=(2,2))(input)
            x2=KL.MaxPooling2D(pool_size=(4,4))(input)
            x3=KL.MaxPooling2D(pool_size=(8,8))(input)
            x4=KL.MaxPooling2D(pool_size=(16,16))(input)
            x1=KL.Conv2D(int(filter/4),(1,1))(x1)
            x2=KL.Conv2D(int(filter/4),(1,1))(x2)
            x3=KL.Conv2D(int(filter/4),(1,1))(x3)
            x4=KL.Conv2D(int(filter/4),(1,1))(x4)
            x1=KL.UpSampling2D(size=(2,2))(x1)
            x2=KL.UpSampling2D(size=(4,4))(x2)
            x3=KL.UpSampling2D(size=(8,8))(x3)
            x4=KL.UpSampling2D(size=(16,16))(x4)
            x=KL.Concatenate()([x1,x2,x3,x4,input])
            x=KL.Conv2D(filter,(1,1))(x)
            return x
   
        def combine(input1,input2,filter):
            x=KL.Activation('relu')(input1)
            x=KL.Concatenate()([x,input2])
            x=KL.Conv2D(filter,(1,1))(x)
            return x
        inputs=KM.Input(shape=(self.config.IMAGE_H, self.config.IMAGE_W, self.config.IMAGE_C))
        c1=x=KL.Conv2D(32,(1,1),strides=(1,1),dilation_rate=1)(inputs)
        c2=x=ResBlock(x,32,(3,3),[1,3,15,31],(1,1))
        x=KL.Conv2D(64,(1,1),strides=(2,2))(x)
        c3=x=ResBlock(x,64,(3,3),[1,3,15,31],(1,1))
        x=KL.Conv2D(128,(1,1),strides=(2,2))(x)
        c4=x=ResBlock(x,128,(3,3),[1,3,15],(1,1))
        x=KL.Conv2D(256,(1,1),strides=(2,2))(x)
        c5=x=ResBlock(x,256,(3,3),[1,3,15],(1,1))
        x=KL.Conv2D(512,(1,1),strides=(2,2))(x)
        c6=x=ResBlock(x,512,(3,3),[1],(1,1))
        x=KL.Conv2D(1024,(1,1),strides=(2,2))(x)
        x=ResBlock(x,1024,(3,3),[1],(1,1))
        x=PSPPooling(x,1024)
        x=KL.Conv2D(512,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c6,512)
        x=ResBlock(x,512,(3,3),[1],1)
        x=KL.Conv2D(256,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c5,256)
        x=ResBlock(x,256,(3,3),[1,3,15],1)
        x=KL.Conv2D(128,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c4,128)
        x=ResBlock(x,128,(3,3),[1,3,15],1)
        x=KL.Conv2D(64,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c3,64)
        x=ResBlock(x,64,(3,3),[1,3,15,31],1)
        x=KL.Conv2D(32,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c2,32)
        x=ResBlock(x,32,(3,3),[1,3,15,31],1)
        x=combine(x,c1,32)
        x=PSPPooling(x,32)
        x=KL.Conv2D(self.config.CLASSES_NUM,(1,1))(x)
        x=KL.Activation('softmax')(x)
        model=KM.Model(inputs=inputs,outputs=x)
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.8),loss=Tanimoto_loss,metrics=['accuracy'])
        model.summary()
        return model

    def train(self, data_path, model_file, restore_model_file=None):
        model = self.model
        if restore_model_file:
            model.load_weights(restore_model_file)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=model_file,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(model_file+"/Unet{epoch:02d}.h5",
                                            verbose=0, save_weights_only=True),
        ]
            

        train_datasets=utils.DataGenerator_wqw(data_path+"/train/image",data_path+"/train/label",self.config.IMAGE_H,self.config.IMAGE_H,self.config.batch_size,self.config.CLASSES_NUM,self.config)
        val_datasets=utils.DataGenerator_wqw(data_path+"/val/image",data_path+"/val/label",self.config.IMAGE_H,self.config.IMAGE_H,self.config.batch_size,self.config.CLASSES_NUM,self.config)

        print ("the number of train data is", len(train_datasets))  
        print ("the number of val data is", len(val_datasets))
        trainCounts = len(train_datasets)
        valCounts = len(val_datasets)
        model.fit_generator(generator=train_datasets,epochs=self.config.EPOCHS,validation_data=val_datasets,callbacks=callbacks, max_queue_size=10,workers=8,use_multiprocessing=True)
    
    def loadWeight(self,path):
        self.model.load_weights(path)

    def predict(self,img):
        img=img-MEAN
        return self.model.predict(img)
    def visual(self,img,path):
        sub_class=int(pow(self.config.CLASSES_NUM,1/3))
        delta=int(255/sub_class)
        color=[]
        curcolor=[0,0,0]
        for i in range(sub_class):
            curcolor[0]=curcolor[0]+delta
                for j in range(sub_class):
                    curcolor[1]=curcolor[1]+delta
                        for k in range(sub_class):
                            curcolor[2]=curcolor[2]+delta
                            color.append(curcolor)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                img[y,x]=color[img[y,x,0]]
        cv2.imwrite(path,img)