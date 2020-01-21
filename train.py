from model import UNet

from config import UnetConfig


config=UnetConfig()

unet=UNet(config=config)

unet.train("dataset-postdam","logs")
