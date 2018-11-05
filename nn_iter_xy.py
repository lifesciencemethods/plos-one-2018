from __future__ import absolute_import

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import Image, ImageDraw
import os
import re
import threading
import warnings
import nn_iter

from keras import backend as K
import nn_image

def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x

pat = re.compile("^inject_(?P<x>[0-9.]+)_(?P<y>[0-9.]+)_.*")

class DirectoryIteratorXY(nn_iter.Iterator):

    def __init__(self, mesh, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None,
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.file_size = (299, 299)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        self.class_mode = "categorical"
        self.mesh = mesh

        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = (3,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.nb_sample = 0
        self.filenames = []

        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".png"):
                self.filenames.append(filename)

        self.nb_sample = len(self.filenames)

        self.nb_class = mesh.VECTOR_LENGTH
        self.class_names = []
        for i in range(0, mesh.VECTOR_LENGTH):
            self.class_names.append("n"+str(i))

        super(DirectoryIteratorXY, self).__init__(self.nb_sample, batch_size, shuffle, seed)


    def filename_to_coordinates(self, fname):
        m = pat.match(fname)
        assert(not m is None)
        x = float(m.group("x"))
        y = float(m.group("y"))
        x = x/self.file_size[0]*self.target_size[0]
        y = y/self.file_size[1]*self.target_size[1]
        return (x, y)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(batch_x), self.nb_class), dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]

            img = nn_image.load_img(os.path.join(self.directory, fname),
                        grayscale=(self.color_mode == "grayscale"),
                        target_size=self.target_size)

            x = nn_image.img_to_array(img, dim_ordering=self.dim_ordering)

            p = self.filename_to_coordinates(fname)

            ppoint = [p]
            x = self.image_data_generator.random_transform(x, ppoint)
            p = ppoint[0]

            x = self.image_data_generator.standardize(x)
            x = preprocess_input(x)

            batch_y[i, :] = self.mesh.point_to_vector(float(p[0])/self.target_size[0], float(p[1])/self.target_size[1])

            batch_x[i] = x

            if self.save_to_dir:
                #q = self.mesh.vector_to_point(batch_y[i, :])
                #q = (q[0]*self.target_size[0], q[1]*self.target_size[1])
                #print "q:", q, p

                CR = 2
                img = nn_image.array_to_img(x, self.dim_ordering, scale=True)
                ctx = ImageDraw.Draw(img)
                ctx.ellipse((p[0]-CR, p[1]-CR, p[0]+CR, p[1]+CR), fill = '#00ff00')
                new_fname = "generated_"+str(i)+"_"+fname
                img.save(os.path.join(self.save_to_dir, new_fname))

        return batch_x, batch_y

