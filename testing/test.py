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
from tqdm import tqdm
import difflib


def main(args):
    f = open(args.config)
    cfgs = yaml.load(f)
    f.close()
    cfgs = ET(cfgs)
    test_list = cfgs.TEST_LIST
    image_size = cfgs.IMAGE_SIZE
    charset = cfgs.CHARSET
    weight = cfgs.WEIGHT

    h, w, c = image_size.split(',')
    image_size = (int(h), int(w), int(c))

    with open(charset) as f:
        charset = f.readline().strip('\n')
        f.close()
    nb_classes = len(charset) + 1

    model, *_ = crnn(nb_classes, image_size)
    model.load_weights(weight, by_name=True)

    test_list = open(test_list).readlines()
    line_acc = 0.
    char_acc = 0.
    total_test = 0
    print('start test..')
    for item in tqdm(test_list):
        img_path, label_str = item.strip('\n').split('\t')
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = reshape_to_target(img, image_size)
        if img is None:
            continue
        img = pre_processing(img)
        img = np.expand_dims(img, axis=0)

        prob = model.predict(img)
        result_str = cd(prob, charset)

        # compute str score
        score = difflib.SequenceMatcher(None, result_str, label_str).ratio()
        if score == 1.0:
            line_acc += 1.0
        char_acc += score
        total_test += 1
    print('test done..')
    print('Line-wise acc: {}%'.format((line_acc/total_test)*100))
    print('Char-wise acc: {}%'.format((char_acc/total_test)*100))
