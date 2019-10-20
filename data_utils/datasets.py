import numpy as np
import cv2
import random
import os
from .transform import reshape_to_target, augmentation, pre_processing


class DataLoader:
    def __init__(self, charset):
        self.charset = charset
        # all chars + blank
        self.nb_classes = len(charset) + 1
        self.char_to_id = None
        self.id_to_char = None
        self._char_to_id(charset)
        self.total_train = 0
        self.total_val = 0

    def _char_to_id(self, charset):
        # insert a special symbol for ctc blank
        self.charset = self.charset + '卍'
        self.char_to_id = {j: i for i, j in enumerate(self.charset)}
        self.id_to_char = {i: j for i, j in enumerate(self.charset)}

    def batch_generator(self, file_list, batch_size=32, max_label_length=40, image_size=(32, 280, 3), training=False):
        with open(file_list) as f:
            file_list = f.readlines()
            f.close()
        h, w, c = image_size
        images = np.zeros((batch_size, h, w, c), dtype=np.float32)
        labels = np.ones([batch_size, max_label_length]) * self.nb_classes
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        while True:
            random.shuffle(file_list)
            for i in range(len(file_list) // batch_size):
                cur_batch = 0
                while cur_batch < batch_size:
                    index = i * batch_size + cur_batch
                    image = file_list[index].split('\t')[0]
                    label = file_list[index].split('\t')[1].strip('\n')
                    if len(label) > max_label_length:
                        continue
                    image = cv2.imread(image)
                    if image is None:
                        continue
                    # resize without change aspect ratio
                    image = reshape_to_target(image, image_size)
                    if image is None:
                        continue
                    # augmentation for training
                    if training:
                        image = augmentation(image)
                    # pre_processing
                    image = pre_processing(image)
                    images[cur_batch] = image
                    labels[cur_batch, :len(label)] = [self.char_to_id[i] for i in label]
                    label_length[cur_batch] = len(label)
                    # hard code
                    input_length[cur_batch] = image_size[1] // 4
                    cur_batch += 1
                inputs = {'the_input': images,
                          'the_labels': labels,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros([batch_size])}
                yield (inputs, outputs)
