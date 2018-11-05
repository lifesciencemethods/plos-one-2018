![Example](https://github.com/lifesciencemethods/plos-one-2018/blob/master/datasets/xy/train/inject_102_130_dpl1496394668.72_train.png)

# PLOS-ONE-2018
Software supplement for PONE-D-18-22290, Maria Lorena Cordero-Maldonado, Simon Perathoner, Kees-Jan van der Kolk, Ralf Boland, Ursula Heins-Marroquin, Herman P. Spaink, Annemarie H. Meijer, Alexander D. Crawford, Jan de Sonneville, “Deep learning image recognition enables efficient genome editing in zebrafish by automated injections”.

## REQUIRED SOFTWARE

Install CUDA (we used version 8.0)
Make sure you have python 2.7 or higher (we used version 2.7.12)
And make sure you have pip (we used version 9.0.1)

Then install virtualenv:

```# sudo pip install virtualenv```

Make sure you are in the root of the package (where this README is located).
Then run:

```# virtualenv main```

Make sure you are running bash as your shell, and activate the virtual environment:

```
# bash
# source main/bin/activate
```

Then install the required packages:

```
# pip install Pillow==3.1.2
# pip install matplotlib==2.0.0
# pip install numpy==1.11.0
# pip install scipy==0.17.0
# pip install six==1.10.0
# pip install PyYAML==3.12
# pip install Theano==0.9.0
# pip install Keras==1.2.2
```

Make sure the following lines are in your $HOME/.theanorc file:

```
    [global]
    device = gpu0
    floatX = float32
```

Make sure the following is in your $HOME/.keras/keras.json file:

```
    {
        "image_dim_ordering": "th",
        "epsilon": 1e-07,
        "floatx": "float32",
        "backend": "theano"
    }
```

And you may wish to set or check the following environment variables for CUDA (these values may be different on your platform):

```
# export CUDA_HOME=/usr/local/cuda
# export CUDA_VISIBLE_DEVICES="0"
# export CUDA_ROOT=/usr/local/cuda/bin/
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
```

## INSTALLATION

This package needs no installation. You can run the Python scripts directly, but please do so in the 
root of the package tree. In particular, the scripts assume that there is a "datasets" directory in the
current directory, containing the datasets; and the scripts assume there is an "output" directory, where
the models will be stored.

Every time you wish to use this software, make sure you are in the root directory of this package, and you are running bash 
as your shell, and always activate the virtual environment:

```
# bash
source main/bin/activate
```


## EGG DETECTION

In order to train the network that is responsible for detecting an egg, use the following command:

```# ./train.py --adam --lr=0.0001 --blocks=11 --batch-size=64 detect```

You can stop the program after a few epochs, when the validation accuracy (val_acc) reaches 1, or is close to 1.

The model is saved after every epoch (as long as the model improves), and therefore stopping the program will not destroy the result.

To validate the model, use:

```# ./predict.py detect datasets/detect/validation/*/*```


## DETERMINING THE INJECTION POSITION

For training the prediction of the injection position, use first the following command:

```# ./train_xy.py --adam --lr=0.001 xy```

Then, when training progress halts (after 10 or 20 epochs), stop the program, and restart with a lower learning rate:

```# ./train_xy.py --adam --lr=0.0001 --continue xy```

Again, stop the program after a few epochs.

To validate the model, use:

```# ./predict_xy.py xy datasets/xy/validation/*```

And view the histogram in the file distance.png




