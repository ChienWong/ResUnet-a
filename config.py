import os
import numpy as np
class UnetConfig(object):
    MEAN=np.array([82,92,88],dtype=float)
    CLASSES_NUM = 6
    IMAGE_W = 512 # image width
    IMAGE_H = 512 # image height
    IMAGE_C = 3 # image channels

    EPOCHS = 5000
    batch_size = 1 


    def displayConfiguration(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

