#!/usr/bin/env python

import sys
import os
import json
import time
import re
import types
from PIL import Image

import numpy as np

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

from nn_image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.utils import np_utils
import keras.callbacks

import nn_net
import nn_image
import nn_iter


file_size = (299, 299)
target_size = (224, 224)


class Predictor:
    def __init__(self, model_file_prefix, strong_augmentation = False):
        if strong_augmentation:
            self.datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=180,
                width_shift_range=0.125,
                height_shift_range=0.125,
                horizontal_flip=False,
                vertical_flip=False,
                zoom_range=0.1,
                channel_shift_range=0,
                fill_mode='wrap')
        else:
            self.datagen = ImageDataGenerator()

        self.model, self.class_names, _ = nn_net.load(model_file_prefix)


    def predict(self, img_or_path, N=1):

        img = img_or_path

        if isinstance(img, types.StringType):
            img = nn_image.load_img(img_or_path, grayscale=False, target_size=target_size)

        if isinstance(img, Image.Image):
            img = nn_image.img_to_array(img, "th")

        time0 = time.time()

        input = []

        for i in range(0, N):
            img2 = self.datagen.random_transform(img)
            img2 = self.datagen.standardize(img2)
            img2 = nn_iter.preprocess_input(img2)

            if 0:
                img3 = nn_image.array_to_img(img2, "th", scale=True)
                img3.save("pred_"+str(i)+".png")

            input.append(img2)

        input = np.stack(input, axis=0)

        vs = self.model.predict(input, batch_size=N)

        v = np.sum(vs, axis=0)
        c = np.argmax(v)

        return self.class_names[c]


if __name__ == "__main__":
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    np.random.seed(1337)

    model_file_prefix = "output/"+sys.argv[1]

    image_paths = sys.argv[2:]

    predictor = Predictor(model_file_prefix)

    pat = re.compile("^(?P<cls>[a-zA-Z]+)_.*")

    M = np.zeros((len(predictor.class_names), len(predictor.class_names)), dtype=int)

    C = {}
    for i,cls in enumerate(predictor.class_names):
        C[cls] = i

    for i,image_path in enumerate(image_paths):

        time0 = time.time()

        m = pat.match(os.path.basename(image_path))
        cls0 = m.group("cls")

        cls1 = predictor.predict(image_path)

        i0 = C[cls0]
        i1 = C[cls1]

        M[i0,i1] += 1

        print "time:", time.time()-time0, "seconds"

        print "result:", cls0, "==> predicted", cls1

        print M




