    
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

Adding the and embeds test data.
The test data is embedded using the weights of the final dense layer, just
before the classification head. This embedding can then be visualized using
TensorBoard's Embedding Projector.
'''
from __future__ import print_function
import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K 
from keras.utils.vis_utils import plot_model

# Tensor Board
from os import makedirs
from os.path import exists, join
from keras.callbacks import TensorBoard
import numpy as np
 
from TrainValTensorBoard import TrainValTensorBoard 
 


# tensorBoard
log_dir = './logs/'

if not exists(log_dir):
    makedirs(log_dir)



batch_size = 128
num_classes = 10
epochs = 12  # default 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# the data is a 2d array
# x_train is the training data set.
# y_train is the set of labels to all the data in x_train.

(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print (x_train.dtype)

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


## tensorBaord
# save class labels to disk to color data points in TensorBoard accordingly
with open(join(log_dir, 'metadata.tsv'), 'w') as f:
    np.savetxt(f, y_test)

## 


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## tensorBaord
tensorboard = TensorBoard(batch_size=batch_size,
                          embeddings_freq=1,
                          embeddings_layer_names=['features'],
                          embeddings_metadata='metadata.tsv',
                          embeddings_data=x_test)

## 

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape));
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu', name='features'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

 

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
cnn_cfir_10 =  model.fit(x_train, y_train,
          batch_size=batch_size,
          callbacks= [TrainValTensorBoard(write_graph=True)],  # [tensorboard,TrainValTensorBoard(write_graph=False)],
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

loss = cnn_cfir_10.history['loss'];
val_loss = cnn_cfir_10.history['val_loss'];


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(model.summary())


# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True);

# print("Hello")