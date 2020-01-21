#ResUnet-a
Semantic segmentation for remote sensing image
###Paper address [https://arxiv.org/abs/1904.00592]
###The original author's implementation of mxnet [https://github.com/feevos/reseta]
#Precondition
#### keras==2.2.2
#### tensorflow==1.9.0
####CV2
####Numpy
#How to train
####Modify the first parameter of unet.train in train.py as the data set address and the second parameter as the model storage address
####See [https://github.com/mohuazheliu/resunet-a/blob/master/dataset-postdam/train/readme.md] for the format of dataset file
#How to Predict
####Refer to test.py, use model.predict to predict a picture, and use model.visual to visualize the prediction
###Nuance
####The implementation method of pspooling is slightly different. Here we refer to the method of pspnet
##Training loss, test results and subsequent update of pre training model
