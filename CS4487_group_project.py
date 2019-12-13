#!/usr/bin/env python
# coding: utf-8

# # CS4487 Machine Learning | City University of Hong Kong
# ## Group Assignment - until 2019.12.13
# - Tim Löhr       , EID: 40126684 
# - Timo Bohnstedt , EID:
# - Karina         , EID:
# 
# ## 1.0 - Preprocessing

# In[11]:


# Keras dependencies
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler

# Other usefull packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ### 1.1 - Load the dataset and set the labels

# In[2]:


classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# ### 1.2 - Visualization of the pictures

# In[3]:


fig = plt.figure(figsize=(20, 4))

for i in np.arange(20):
    ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    plt.imshow(x_train[:20][i], cmap='gray')
    ax.set_title(classes[y_train[:20][i].item()])


# In[4]:


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate


# ### 1.3 - Data manipulatiom

# In[5]:


trainXs = stats.zscore(x_train)
testXs = stats.zscore(x_test)


# In[6]:


#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

#z-score
#mean = np.mean(x_train,axis=(0,1,2,3))
#std = np.std(x_train,axis=(0,1,2,3))
#x_train = (x_train-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)


# Converts our input class vector (50000, 1) to a binary output class matrix (50000, 10)
num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)


# In[7]:


# Shapes
print('Train X Shape: ' + str(trainXs.shape))
print('Train y Shape: ' + str(y_train.shape))


# In[8]:


#data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.125,
    height_shift_range=0.125,
    fill_mode='constant',
    cval=0.
    )

datagen.fit(trainXs)


# ## 2.0 - Model

# In[9]:


weight_decay = 1e-4


# In[18]:


model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# ## 3.0 Model Training

# ### 3.1 - Fit model 

# In[19]:


#training
batch_size = 64

sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=500, 
                    epochs=150,
                    verbose=1,
                    validation_data=(x_test,y_test),
                    callbacks=[LearningRateScheduler(lr_schedule)])


# ### 3.2 - Save model

# In[ ]:


#save to disk
model_json = model.to_json()
with open('model_timo.json', 'w') as json_file:
    json_file.write(model_json)



model.save_weights('model_timo.h5')    

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

