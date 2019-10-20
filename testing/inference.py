import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
K.set_session(sess)

import os
import sys

sys.path.insert(0, '../')
from models.crnn import crnn
from data_utils.transform import reshape_to_target, pre_processing
from .ctc_decode import ctc_decode as cd

import yaml
import cv2
import numpy as np
from easydict import EasyDict as ET


def main(args):
    f = open(args.config)
    cfgs = yaml.load(f)
    f.close()
    cfgs = ET(cfgs)
    image_size = cfgs.IMAGE_SIZE
    charset = cfgs.CHARSET
    weight = cfgs.WEIGHT
    img_path = args.img

    h, w, c = image_size.split(',')
    image_size = (int(h), int(w), int(c))

    with open(charset) as f:
        charset = f.readline().strip('\n')
        f.close()
    nb_classes = len(charset) + 1

    model, *_ = crnn(nb_classes, image_size)
    model.load_weights(weight, by_name=True)

    ori_img = cv2.imread(img_path)
    img = reshape_to_target(ori_img, image_size)
    img = pre_processing(img)
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)
    result_str = cd(prob, charset)
    print(result_str)
    if args.viz:
        cv2.imshow('display', ori_img)
        cv2.waitKey(0)
