#!/usr/bin/env python

import sys
import json
import time
import getopt

import numpy as np
from collections import defaultdict

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

from nn_image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.utils import np_utils
import keras.callbacks

import nn_net
import nn_iter_xy
import nn_mesh

mesh = nn_mesh

np.random.seed(1337)

batch_size = 128
image_size = (224, 224)
progress_threshold = 20
initial_learning_rate = 0.0001
num_blocks = 2
strong_augmentation = True

opts,args = getopt.getopt(sys.argv[1:], "", ["continue", "batch-size=", "lr=", "blocks=", "adam", "reset"])


should_continue = False
should_reset = False
use_optimizer = "sgd"

for (k,v) in opts:
    if k == "--adam":
        initial_learning_rate = 0.001 # This seems optimal for Adam

for (k,v) in opts:
    if k == "--continue":
        should_continue = True
    if k == "--lr":
        initial_learning_rate = float(v)
    if k == "--blocks":
        num_blocks = int(v)
    if k == "--batch-size":
        batch_size = int(v)
    if k == "--adam":
        use_optimizer = "adam"
    if k == "--reset":
        should_reset = True

model_name = args[0]

data_directory = "datasets/"+model_name
model_file_prefix = "output/"+model_name

if strong_augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=180,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.1,
        channel_shift_range=0,
        fill_mode='wrap')
else:
    datagen = ImageDataGenerator()

train_generator = nn_iter_xy.DirectoryIteratorXY(mesh, data_directory+"/train", datagen, target_size=image_size, batch_size=batch_size)
validation_generator = nn_iter_xy.DirectoryIteratorXY(mesh, data_directory+"/validation", datagen, target_size=image_size, batch_size=batch_size)

history = []
last_switch_epoch = 0

class LoggingCallback(keras.callbacks.Callback):
    def __init__(self):
        pass
        
    def on_epoch_end(self, epoch, logs={}):
        global last_switch_epoch

        logs["description"] = model.description
        logs["time"] = time.time()
        logs["args"] = sys.argv[1:]

        def tolist(l, attr):
            r = []
            for m in l:
                r.append(m.get(attr))
            return r

        history.append(logs)
        L = tolist(history, "val_acc")
        max_val_acc = max(L)

        if(logs["val_acc"] >= max_val_acc):
            nn_net.save(model, train_generator.class_names, model_file_prefix, history)
            #print "[saved]"

        with open(model_file_prefix+"-log.json", "w") as json_file:
            json.dump(history, json_file)

        if np.argmax(L[last_switch_epoch:]) < len(L[last_switch_epoch:])-progress_threshold:
            print >>sys.stderr, ""
            print >>sys.stderr, "No progress, stopping"
            print >>sys.stderr, ""
            last_switch_epoch = len(L)
            model.stop_training = True


if not should_continue:
    model = nn_net.build_model(len(train_generator.class_names))
else:
    print "Loading model from file"
    model, _, history = nn_net.load(model_file_prefix)

BLOCKS_TO_LAYERS = [217, 195, 172, 159, 137, 115, 93, 71, 61, 45, 29, 0]

num_layers = int(BLOCKS_TO_LAYERS[num_blocks])

for layer in model.layers[:num_layers]:
   layer.trainable = False
for layer in model.layers[num_layers:]:
   layer.trainable = True

if should_reset:
    nn_net.reset_trainable_layers(model)

def adjust_learning_rate(factor = 0.5):
    model.current_learning_rate *= float(factor)
    model.description = use_optimizer+" lr="+str(model.current_learning_rate)
    print "Setting learning rate to", model.current_learning_rate, use_optimizer
    if use_optimizer == "sgd":
        model.compile(optimizer=SGD(lr=model.current_learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])
    elif use_optimizer == "adam":
        model.compile(optimizer = Adam(lr=model.current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=["accuracy"])
    else:
        assert(0)

model.current_learning_rate = initial_learning_rate
model.adjust_learning_rate = adjust_learning_rate

model.adjust_learning_rate(1.0)

while True:
    model.fit_generator(train_generator,
                samples_per_epoch=train_generator.nb_sample*30, # Note: we multiply by 30 because the dataset is small.
                nb_epoch=1000000,
                validation_data=validation_generator,
                nb_val_samples=validation_generator.nb_sample,
                callbacks=[LoggingCallback()]
                )
    print >>sys.stderr, "Restarting with different learning rate"
    model.adjust_learning_rate()

