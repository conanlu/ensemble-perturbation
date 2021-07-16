from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from utils import utils
from advfaces import AdvFaces
import cv2
import os
import scipy.misc
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

import numpy as np
# Load Adversarial Face Generator
network = AdvFaces()
network.load_model('pretrained/obfuscation')

# 2. Folder of images
dataset = Dataset('data')

# Load config and images
config = utils.import_file('config/default.py', 'config')
images = preprocess(dataset.images, config, is_training=False)

# Generate Adversarial Images and Adversarial Masks


adversaries, adversaries_mask = network.generate_images(images)
for i, adv in enumerate(adversaries):
    print("adversary ", type(adversaries[i]))

    adversaries[i] = images[i] + adversaries_mask[i]
    print("adversaries")
    print(adversaries[i][155, 155])

    print("images")
    print(images[i][155, 155])

    print("adversaries_mask")



    mean = 127.5
    std = 128.0

    adversarytodisplay = adversaries[i].copy()
    adversarytodisplay = (adversarytodisplay * std) + mean
    adversarytodisplay = np.clip(adversarytodisplay, 0, 255)
    advers = adversarytodisplay.astype(np.uint8)

    r,g,b = cv2.split(advers)
    advers = cv2.merge((b,g,r))


    cv2.imshow("yayyy", advers)
    cv2.waitKey(0)

    r, g, b = cv2.split(advers)
    advers = cv2.merge((b, g, r))


    print(adversaries_mask[i][155, 155])



    #make sure you edit the i number 0001 correctly
    stri = str(i+1)
    if(i < 10):
        stri = "0000"+stri
    elif(i < 100):
        stri = "000"+stri
    elif(i < 1000):
        stri = "00"+stri
    elif(i < 10000):
        stri = "0"+stri
    scipy.misc.imsave('results/{}.jpg'.format(stri), advers)
    exit(0)

# # Save adversarial image
# scipy.misc.imsave('results/result.jpg', adversaries[0])
