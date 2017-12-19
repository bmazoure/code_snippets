import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
num_classes = 10
epochs = 2
trials = 5
train_size  = 10000
eps = 1e-5
# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:train_size]
y_train = y_train[:train_size]

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
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def get_model(loss,uncertain=False):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if uncertain:
        model.add(Lambda(lambda x: K.dropout(x, level=0.25)))
    else:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if uncertain:
        model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
    else:
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
#%%
model1 = get_model(keras.losses.categorical_crossentropy,uncertain=True)
model1.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
#%%
def get_weights(x,trials,f):
    y_preds = []
    for trial in range(trials):
        y_preds.append(model1.predict(x))
    y_preds = np.array(y_preds)
    uncertainty = np.max(np.var(y_preds,axis=0),axis=1)
    weights = f(uncertainty)
    plt.hist(weights,bins=50)
    weights = np.dstack([weights]*10)[0] # Repeat same thing for each classs
    return weights
f = lambda x: np.exp(-np.power(x,1./4.))
weights = get_weights(x_train,trials,f)
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
#%%
batch_weights = batch(weights,batch_size)
def weighted_categorical_crossentropy(weights):
    #weights = K.variable(weights)      
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        l_weights = next(weights)
    
        loss = y_true * K.log(y_pred) * l_weights
        
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('accuracy'))
        
weights_dummy = np.ones(10)
loss_weighted=weighted_categorical_crossentropy(batch_weights)

history_weighted = LossHistory()
model_weighted = get_model(loss_weighted)
model_weighted.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=False,
          callbacks=[history_weighted])

history_not_weighted = LossHistory()
model_not_weighted = get_model(loss_weighted)
model_not_weighted.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=False,
          callbacks=[history_not_weighted])
x = [x for x in range(len(history_weighted.losses))]
plt.plot(x,history_weighted.losses,'r')
plt.plot(x,history_not_weighted.losses,'b')
plt.show()