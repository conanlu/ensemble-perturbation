from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from utils import utils
from advfaces import AdvFaces
import os
import scipy.misc
import shutil
import math
from PIL import Image
import argparse
import os.path
import sys
import cv2
import dlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from imutils import face_utils
from src.models import inception_resnet_v1
from utils.ThinPlateSpline2 import ThinPlateSpline2 as TPS
from utils.attack_util import purturb_GFLM
import matplotlib
import matplotlib.pyplot as plt
import crop





#FLM METHOD
def image_warping(img, lndA, lndB):
    CROP_SIZE = 182
    input_images_expanded = tf.reshape(img, [1, CROP_SIZE, CROP_SIZE, 3, 1])
    t_img, T, det = TPS(input_images_expanded, lndA, lndB, [CROP_SIZE, CROP_SIZE, 3])
    t_img = tf.reshape(t_img, [1, CROP_SIZE, CROP_SIZE, 3])
    return t_img, T


def prepare_for_save(x):
    x = (x - x.min()) / (x.max() - x.min())
    x = (x * 255.).astype(np.uint8)
    x = x[..., ::-1]
    return x


def mapback(x):
    return (x + 1.) * 182. / 2.


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="FLM", choices=["FLM", "GFLM"])
parser.add_argument("--intensity", type=int, help='number of steps', default=10)
parser.add_argument('--output_dir', type=str,
                    help='Directly where output files will be saved.',
                    default='output')
parser.add_argument('--img', type=str,
                    help='Path to the input image.',
                    default='data/')


args = parser.parse_args()

#pre-settings on FLM you will never touch
seed = 105
fixed_points = 4
label = 58 #what is this
epsilon = 0.005
prelogits_hist_max = 10.0
use_fixed_image_standardization = True
embedding_size = 128
image_size = 182
model_def = 'src.models.inception_resnet_v1'

#locations of things. ignore the slash inconsistencies. i keep you on your toes :)
dlib_model = 'shape_predictor_68_face_landmarks.dat'
pretrained_model = '20180408-102900/'
pretrained_model = '20180408-102900/'
face_detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascades/haarcascade_eye1.xml")

#THINGS YOU WANT TO CHANGE PROBABLY
mode = args.mode #'GFLM' # or GFLM
#numpics = 101
intensity = args.intensity
output_dir = args.output_dir
imgpath = args.img

# Load Adversarial Face Generator
network = AdvFaces()
network.load_model('pretrained/obfuscation')

## Load images
# Images can be loaded via
# 1. Image Filelist
# dataset = Dataset('image_list.txt')

# 2. Folder of images

dat = crop.fix(3)
dataset = Dataset('fixed')



# Load config and images
config = utils.import_file('config/default.py', 'config')
images = preprocess(dataset.images, config, is_training=False)


#Defining FLM

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model)

networkk = inception_resnet_v1
y = tf.placeholder(tf.int32)

# define the source and target landmark locations
lnd_A = tf.placeholder(tf.float32, [None, 2])
lnd_B = tf.placeholder(tf.float32, [None, 2])

x = tf.placeholder(tf.float32, shape=[182, 182, 3])
imagess = x
imagess = tf.image.per_image_standardization(imagess)
imagess = tf.reshape(imagess, [-1, 182, 182, 3])

lnd_source = tf.expand_dims(lnd_A, axis=0)
lnd_target = tf.expand_dims(lnd_B, axis=0)

images_deformed, T = image_warping(imagess, lnd_target, lnd_source)

images_deformed = tf.image.per_image_standardization(images_deformed[0])
images_deformed = tf.expand_dims(images_deformed, axis=0)


if pretrained_model:
    pretrained_model = os.path.expanduser(pretrained_model)
    print('Trained model: %s' % pretrained_model)
else:
    exit("A pretrained model should be provided!")



tf.set_random_seed(seed)

# Build the inference graph
prelogits, cam_conv, _ = networkk.inference(images_deformed, 1.,
                                 phase_train=False, bottleneck_layer_size=512)
logits = slim.fully_connected(prelogits, 10575, activation_fn=None,
                              scope='Logits', reuse=False)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
softmax = tf.nn.softmax(logits)
grad = tf.gradients(loss, lnd_B)[0]*1.

# Create a saver
saver = tf.train.Saver()


with tf.Session() as sess:
    print("trying to load model")
    saver = tf.train.import_meta_graph(pretrained_model + 'model-20180408-102900.meta')
    saver.restore(sess, pretrained_model + 'model-20180408-102900.ckpt-90')
    print("done loading model")

    adversariesold = images
    adversariesnew = adversariesold

    for n in np.arange(2-1):
        # print(np.shape(adversariesold))
        # print(np.shape(images))
        # network.generate_images(adversariesold)
        adversariesnew, adversariesnew_mask = network.generate_images(adversariesold)
        for iii, adv in enumerate(adversariesnew):
            adversariesnew[iii] = adversariesold[iii] + adversariesnew_mask[iii]
        adversariesold = adversariesnew

    adversaries, adversaries_mask = network.generate_images(adversariesold)

    for i, adv in enumerate(adversaries):

        print(i)

        #finding file that corresponds with i in the dataset
        stri = str(i+1)
        if (i+1 < 10):
            stri = "0000" + stri
        elif (i+1 < 100):
            stri = "000" + stri
        elif (i+1 < 1000):
            stri = "00" + stri
        elif (i+1 < 10000):
            stri = "0" + stri


        img = cv2.imread(imgpath + stri + ".jpg")

        # convert color image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # extract landmarks
        try:
            rect = detector(gray, 1)[0]
        except IndexError as e:
            print("face not detected at {}, skipping", i)
            continue
        rect = detector(gray, 1)[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #thing = cv2.resize(adversaries_mask[i], (182, 182))
        #images[i] = cv2.resize(images[i], (182, 182))

        adversaries[i]= adversariesold[i] + adversaries_mask[i]
        mean = 127.5
        std = 128.0

        adversarytodisplay = adversaries[i].copy()
        adversarytodisplay = (adversarytodisplay * std) + mean
        adversarytodisplay = np.clip(adversarytodisplay, 0, 255)
        advers = adversarytodisplay.astype(np.uint8)

        r, g, b = cv2.split(advers)
        advers = cv2.merge((b, g, r))




        img = cv2.resize(advers, (182, 182))



        # generate edge points to force the transformation to keep the boundary
        step = fixed_points
        #print("step ", step)
        new_w = 182
        steps = np.array(list(range(0, new_w, new_w//step)))
        #print("steps ", steps)
        b = list()
        for s in steps:
            b.append([0, s])
            b.append([s, 0])
            b.append([182, s])
            b.append([s, 182])
        b = np.array(b)
        b = b.reshape([-1, 2])
        shape = np.concatenate((shape, b), axis=0)
        lnd = np.copy(shape)

        # convert to rgb
        img = img[..., ::-1]

        # remove mean and scale pixel values
        img = (img - 127.5) / 128.

        # scale landmarks to [-1, 1]
        dp = np.copy((lnd / 182.) * 2. - 1.)
        lnd = np.copy(dp)

        # initialize the landmark locations of the adversarial face image
        lnd_adv = np.copy(lnd)
        #print("lnd shape ", lnd.shape)

        # print ('True label:', args.label)

        if not os.path.exists(os.path.join(output_dir, "incremental")):
            os.makedirs(os.path.join(output_dir, "incremental"))

        # print ('True label:', args.label)
        for ii in range(intensity):
            print("ii ", ii)
            l, s, img_d, t, grad_ = sess.run([logits, softmax, images_deformed, T, grad],
                                             feed_dict={x: img, lnd_A: lnd, lnd_B: lnd_adv, y: [label]})
            #print("step: %02d, Predicted class: %05d, Pr(predicted class): %.4f, Pr(true class): %.4f" %
                  #(ii, np.argmax(l), s.max(), s[0, label]))

            # print("argmax ", np.argmax(l), " args.label ", args.label)
            # if np.argmax(l) != args.label:
            #     print("break")
            #     break
            epsilon = epsilon

            if mode == "GFLM":
                lnd_adv = purturb_GFLM(lnd_adv, grad=grad_, epsilon=epsilon)
            else:
                lnd_adv = lnd_adv + np.sign(grad_) * epsilon

            # if (ii % 5) == 0:
            #     #print(ii%5)
            #



        temp = prepare_for_save(img_d.reshape([182, 182, 3]))
        cv2.imwrite(output_dir+"/"+stri+".jpg", temp)

    crop.reattach(dat)



