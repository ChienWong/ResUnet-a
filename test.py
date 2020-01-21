from model import UNet

from config import UnetConfig
import cv2

config=UnetConfig()

unet=UNet(config=config)

unet.loadWeight("./logs")
imgdir=os.listdir("./dataset-postdam/test")
index=0
for name in imgdir:
    img=cv2.imread("./dataset-postdam/test/"+name)
    img=cv2.resize(img,(config.IMAGE_H,config.IMAGE_W))
    result=unet.predict(img)
    unet.visual(result,"./test-result/"+str(index)+".png")
    index=index+1
