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
import nn_mesh
import nn_iter_xy

mesh = nn_mesh

file_size = (299, 299)
target_size = (224, 224)


def combine_points(ps, cutoff = .4):

    def average(ps):
        x = 0.0
        y = 0.0
        for p in ps:
            x += p[0]
            y += p[1]
        x /= len(ps)
        y /= len(ps)
        return (x, y)

    avg = average(ps)

    def cmp(p0, p1):
        x0 = avg[0]-p0[0]
        y0 = avg[1]-p0[1]
        x1 = avg[0]-p1[0]
        y1 = avg[1]-p1[1]
        d0 = x0*x0+y0+y0
        d1 = x1*x1+y1+y1
        if d0 < d1: return 1
        if d1 < d0: return -1
        return 0

    ps.sort(cmp)

    ps = ps[int(cutoff*len(ps)):]

    return average(ps)



class PredictorXY:
    def __init__(self, model_file_prefix, strong_augmentation = True):
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

        self.model, _, _ = nn_net.load(model_file_prefix)


    def predict(self, img_or_path, N=1): # Returns coordinates in range from (0,0) to target_size.

        img = img_or_path

        if isinstance(img, types.StringType):
            img = nn_image.load_img(img_or_path, grayscale=False, target_size=target_size)

        if isinstance(img, Image.Image):
            img = nn_image.img_to_array(img, "th")

        time0 = time.time()

        input = []
        transform_back_functions = []

        for i in range(0, N):
            p_transform_back = [0]
            img2 = self.datagen.random_transform(img, p_transform_back = p_transform_back)
            img2 = self.datagen.standardize(img2)
            img2 = nn_iter_xy.preprocess_input(img2)

            if 0:
                img3 = nn_image.array_to_img(img2, "th", scale=True)
                img3.save("pred_"+str(i)+".png")

            input.append(img2)
            transform_back_functions.append(p_transform_back[0])

        #print "transform-time:", time.time()-time0

        input = np.stack(input, axis=0)

        vs = self.model.predict(input, batch_size=N)

        ps = []

        for i in range(0, vs.shape[0]):
            v = vs[i, :]
            #print "v=", v
            p = mesh.vector_to_point(v)
            p = (p[0]*target_size[0], p[1]*target_size[1])
            p = transform_back_functions[i](p)
            #print "p=", p
            ps.append(p)

        return combine_points(ps)


if __name__ == "__main__":
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    np.random.seed(1337)

    model_file_prefix = "output/"+sys.argv[1]

    image_paths = sys.argv[2:]

    predictor_xy = PredictorXY(model_file_prefix)

    pat = re.compile("^inject_(?P<x>[0-9.]+)_(?P<y>[0-9.]+)_.*")
    distances = []

    def make_plot():
        print >>sys.stderr, "plotting ..."

        plt.figure()
        n, bins, patches = plt.hist(distances, 10, normed=0, facecolor='green', alpha=0.75)

        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.xlim([0, 160])
        plt.grid(True)
        plt.savefig("distance.png");
        plt.savefig("distance.eps");

    plot_time = time.time()
    total_time = 0
    total_n = 0

    for i,image_path in enumerate(image_paths):

        time0 = time.time()
        if time0 - plot_time > 20:
            make_plot()
            plot_time = time0

        time0 = time.time()

        print str(int(100*float(i)/len(image_paths)))+"%"

        m = pat.match(os.path.basename(image_path))
        x = float(m.group("x"))
        y = float(m.group("y"))
        expected = (x, y)

        print image_path, expected

        r = predictor_xy.predict(image_path)
        r = (r[0]/target_size[0]*file_size[0], r[1]/target_size[1]*file_size[1])
        #r = (201, 145)

        dx = r[0]-expected[0]
        dy = r[1]-expected[1]

        d = (dx*dx+dy*dy)**.5
        distances.append(d)

        if i != 0: # First prediction takes much longer, so don't include it in statistics.
            elapsed = time.time()-time0
            total_time += float(elapsed)
            print "time:", elapsed, "seconds"
            total_n += 1
            print "avg distance:", float(sum(distances))/len(distances), " avg time:", total_time/total_n, "seconds"

        print "result:", r, "..", expected, "==", d


    make_plot()

    with open("distance.csv", "w") as fp:
        for d in distances:
            print >>fp, d


