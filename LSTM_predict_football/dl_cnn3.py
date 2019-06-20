# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 08:23:59 2018

@author: Administrator
"""


from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten



from keras.layers import Conv2D,MaxPooling2D,Input
from keras.models import Model
from keras.layers import *

import os
import dl_data_process as dp



batch_size = 32
n_classes = 3
epochs = 20
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_model(8.17_cnn(1.5)).h5'

file =  'odds_4_cnn(1.5).dat'
# The data, split between train and test sets.
(x_train_1, x_train_2, y_train), (x_test_1, x_test_2, y_test) = dp.load_data_cnn(file)

x_train_1 = x_train_1.reshape(x_train_1.shape[0], x_train_1.shape[1], x_train_1.shape[2], 1)
x_train_2 = x_train_2.reshape(x_train_2.shape[0], x_train_2.shape[1], x_train_2.shape[2], 1)
x_test_1 = x_test_1.reshape(x_test_1.shape[0], x_test_1.shape[1], x_test_1.shape[2], 1)
x_test_2 = x_test_2.reshape(x_test_2.shape[0], x_test_2.shape[1], x_test_2.shape[2], 1)



# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)



# First, let's define a odds model using a Sequential model.
# This model will encode an image into a vector.
odds_model = Sequential()
odds_model.add(Conv2D(64, (2, 2), activation='relu', padding='same', input_shape=(x_train_1.shape[1], x_train_1.shape[2], 1)))
odds_model.add(MaxPooling2D((2, 2)))
odds_model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
odds_model.add(MaxPooling2D((2, 2)))
odds_model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
odds_model.add(Dropout(0.25))
odds_model.add(Dense(512))
#odds_model.add(Dense(n_classes, activation='softmax'))
odds_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
odds_input = Input(shape=(x_train_1.shape[1], x_train_1.shape[2], 1))
odds_output = odds_model(odds_input)



kelly_model = Sequential()
kelly_model.add(Conv2D(64, (2, 2), activation='relu', padding='same', input_shape=(x_train_1.shape[1], x_train_1.shape[2], 1)))
kelly_model.add(MaxPooling2D((2, 2)))
kelly_model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
kelly_model.add(MaxPooling2D((2, 2)))
kelly_model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
kelly_model.add(Dropout(0.25))
kelly_model.add(Dense(512))
#kelly_model.add(Dense(n_classes, activation='softmax'))
kelly_model.add(Flatten())

kelly_input = Input(shape=(x_train_2.shape[1], x_train_2.shape[2], 1))
kelly_output = kelly_model(kelly_input)


merged = keras.layers.concatenate([odds_output, kelly_output], axis=1)

main_output = Dense(3, activation='softmax', name='main_output')(merged)

model = Model(inputs=[odds_input, kelly_input], outputs= main_output)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#model.fit([x_train_1, x_train_2], y_train, batch_size=32, epochs=10,verbose=1)

model.fit([x_train_1, x_train_2], y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([x_test_1,  x_test_2], y_test),
          shuffle=True)

score = model.evaluate([x_test_1,  x_test_2], y_test, batch_size=10, verbose=1)
print(score)


y_pre = model.predict([x_test_1,  x_test_2])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


'''

#construct model
odds_input = Input(( x_train_1.shape[1], x_train_1.shape[2], 1), dtype='float32', name='odds_input' )
x_odds = Conv2D(32,(2,2),padding='same',activation='relu')(odds_input)
x_odds = Dropout(0.25)(x_odds)
x_odds = Conv2D(64,(2,2),padding='same',activation='relu')(x_odds) 
x_odds = Dropout(0.25)(x_odds)
x_odds = Conv2D(64,(2,2),padding='same',activation='relu')(x_odds) 
x_odds = Dropout(0.25)(x_odds)
x_odds = MaxPooling2D((2,2),strides=(1,1),padding='same')(x_odds)
odds_out = Dense(num_classes, activation='softmax')(x_odds)

kelly_input = Input(( x_train_2.shape[1], x_train_2.shape[2], 1), dtype='float32', name='kelly_input' )
x_kelly = Conv2D(32,(2,2),padding='same',activation='relu')(kelly_input)
x_kelly = Dropout(0.25)(x_kelly)
x_kelly = Conv2D(64,(2,2),padding='same',activation='relu')(x_kelly)
x_kelly = Dropout(0.25)(x_kelly)
x_kelly = Conv2D(64,(2,2),padding='same',activation='relu')(x_kelly)
x_kelly = Dropout(0.25)(x_kelly)
x_kelly = MaxPooling2D((2,2),strides=(1,1),padding='same')(x_kelly)
kelly_out = Dense(num_classes, activation='softmax')(x_kelly)


x = keras.layers.concatenate([odds_out, kelly_out], axis=0)
#x = merge([odds_out, kelly_out], mode='concat', concat_axis=2)
main_output = Dense(num_classes, activation='softmax', name='main_output')(x)

model = Model(inputs=[odds_input, kelly_input], outputs=[odds_out, kelly_out, main_output])
model.compile(optimizer='rmsprop', 
            loss={'main_output': 'categorical_crossentropy', 'odds_output': 'categorical_crossentropy','kelly_output': 'categorical_crossentropy'},
            loss_weights={'main_output': 1., 'odds_output': 0.5, 'kelly_output': 0.5})


model.fit(x={'odds_input': x_train_1, 'kelly_input': x_train_2},
            y={'main_output': y_train, 'odds_output': y_train,'kelly_output': y_train},
            batch_size=32, epochs=10,verbose=1)
 
score = model.evaluate(x={'odds_input': x_test_1, 'kelly_input': x_test_2},
            y={'main_output': y_test, 'odds_output': y_test, 'kelly_output': y_test},
            batch_size=10, verbose=1)

print(score)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

'''







