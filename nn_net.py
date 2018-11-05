
import json

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D


# create the base pre-trained model
def build_model(nb_classes):

    null_model = InceptionV3(weights=None, include_top=False)

    base_model = InceptionV3(weights="imagenet", include_top=False)

    for i,layer in enumerate(null_model.layers):
        base_model.layers[i].random_weights = null_model.layers[i].get_weights()
        #print i, base_model.layers[i].random_weights

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)

    # Freeze all Inception layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def reset_trainable_layers(model):
    count = 0
    for layer in model.layers:
        if getattr(layer, "trainable", False):
            w = getattr(layer, "random_weights", None)
            if not w is None:
                layer.set_weights(w)
                count += 1
    print "Reset", count, "layers"


def save(model, tags, prefix, history):
    model.save_weights(prefix+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open(prefix+".json", "w") as json_file:
        json_file.write(model_json)
    with open(prefix+"-labels.json", "w") as json_file:
        json.dump(tags, json_file)
    with open(prefix+"-history.json", "w") as json_file:
        json.dump(history, json_file)


def load(prefix):
    # load json and create model
    with open(prefix+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(prefix+".h5")
    with open(prefix+"-labels.json") as json_file:
        tags = json.load(json_file)
    with open(prefix+"-history.json") as json_file:
        history = json.load(json_file)
    return model, tags, history

