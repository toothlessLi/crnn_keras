import cv2
import numpy as np
import random


def reshape_to_target(img, target_shape):
    h, w, _ = img.shape
    ratio_h = 1.0 * target_shape[0] / h
    scaled_w = int(w * ratio_h)
    if scaled_w > target_shape[1] or scaled_w < target_shape[0]:
        # drop too long or too short sample
        return None
    img = cv2.resize(img, (scaled_w, target_shape[0]))

    pad_left = (target_shape[1] - scaled_w) // 2
    pad_right = target_shape[1] - pad_left - scaled_w
    img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img)
    img -= 128.
    img /= 128.
    return img


def augmentation(img):
    # not implement
    return img
