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
from models.crnn import get_model
from data_utils.datasets import DataLoader
from keras.callbacks import ModelCheckpoint

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as ET


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def main(args):
    f = open(args.config)
    cfgs = yaml.load(f)
    f.close()
    cfgs = ET(cfgs)
    train_list = cfgs.TRAIN_LIST
    val_list = cfgs.VAL_LIST
    image_size = cfgs.IMAGE_SIZE
    charset = cfgs.CHARSET
    max_label_length = cfgs.MAX_LABEL_LENGTH
    save_dir = cfgs.SAVE_DIR
    epochs = cfgs.EPOCHS
    train_batch_size = cfgs.TRAIN_BATCH_SIZE
    val_batch_size = cfgs.VAL_BATCH_SIZE

    h, w, c = image_size.split(',')
    image_size = (int(h), int(w), int(c))

    with open(charset) as f:
        charset = f.readline().strip('\n')
        f.close()
    nb_classes = len(charset) + 1

    model = get_model(nb_classes, image_size, max_label_length)
    data_loader = DataLoader(charset)

    total_train = len(open(train_list).readlines())
    total_val = len(open(val_list).readlines())

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(save_dir + '/weights-{epoch:02d}.h5',
                                 save_weights_only=True)

    res = model.fit_generator(data_loader.batch_generator(train_list, train_batch_size,
                                                          max_label_length, image_size, True),
                              steps_per_epoch=total_train // train_batch_size,
                              epochs=epochs,
                              validation_data=data_loader.batch_generator(val_list, val_batch_size,
                                                                          max_label_length, image_size, False),
                              validation_steps=total_val // val_batch_size,
                              callbacks=[checkpoint],
                              verbose=1)

    plot_history(res, save_dir)
    save_history(res, save_dir)
