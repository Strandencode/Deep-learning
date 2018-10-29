from __future__ import division
import numpy as np
import keras
print('Using Keras version', keras.__version__)

from keras.datasets import cifar10
from keras.optimizers import SGD



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Check sizes of dataset
print('Number of train examples', x_train.shape[0])
print('Size of train examples', x_train.shape[1:])

#Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
print(x_train.shape)

# Convert class vectors to binary class matrices.
num_classes = 10
y_train = np.squeeze(keras.utils.to_categorical(y_train, num_classes))
y_test = np.squeeze(keras.utils.to_categorical(y_test, num_classes))

#Find which format to use (depends on the backend), and compute input_shape
from keras import backend as K
#MNIST resolution
img_rows, img_cols = 32, 32

#Depending on the version of Keras, two different sintaxes are used to specify the ordering
###Keras 1.X (as in MinoTauro)
#K.image_dim_ordering == 'tf' or 'theano'
###Keras 2.X (probably in your local installation)
#K.image_data_format == 'channels_first' or 'channels_last'
#

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

epochs = 40
batch_size = 32
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



#fit the model
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
# Fit the model


#Evaluate the model with test set
score = model.evaluate(x_test, y_test, verbose=1)
print('test loss:', score[0])
print('test accuracy:', score[1])


##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#No validation loss in this example
plt.legend(['train','val'], loc='upper left')
plt.savefig('model_accCNNshallowdrop.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('model_lossCNNshallowdrop.pdf')


#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
Y_pred = model.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)

#Plot statistics
print('Analysis of results')
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
