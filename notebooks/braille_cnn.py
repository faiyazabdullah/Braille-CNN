import os
import numpy as np
import pandas as pd
from shutil import copyfile

from keras import backend as K
from keras import layers as L
from keras.models import Model,load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.layers import Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# import matplotlib.pyplot as plt

os.mkdir('./images/')
alpha = 'a'
for i in range(0, 26): 
    os.mkdir('./images/' + alpha)
    alpha = chr(ord(alpha) + 1)

rootdir = 'dataset/Braille Dataset/Braille Dataset/'
for file in os.listdir(rootdir):
    letter = file[0]
    copyfile(rootdir+file, './images/' + letter + '/' + file)

datagen = ImageDataGenerator(rotation_range=20,
                             shear_range=10,
                             validation_split=0.2)

train_generator = datagen.flow_from_directory('./images/',
                                              target_size=(28,28),
                                              subset='training')

val_generator = datagen.flow_from_directory('./images/',
                                            target_size=(28,28),
                                            subset='validation')

# from tensorflow.keras.preprocessing import image
# img=image.load_img('dataset/Braille Dataset/Braille Dataset/a1.JPG0dim.jpg')

# x=image.img_to_array(img)
# x.shape
# x=np.expand_dims(x,axis=0)
# model.predict(x)

# K.clear_session()

model_ckpt = ModelCheckpoint('BrailleNet.h5', save_best_only=True)
reduce_lr = ReduceLROnPlateau(patience=8, verbose=0)
early_stop = EarlyStopping(patience=15, verbose=1)

entry = L.Input(shape=(28, 28, 3))
x = L.SeparableConv2D(64, (3, 3), activation='relu')(entry)
x = L.BatchNormalization()(x)
x = L.MaxPooling2D((2, 2))(x)

x = L.SeparableConv2D(128, (3, 3), activation='relu')(x)
x = L.BatchNormalization()(x)
x = L.MaxPooling2D((2, 2))(x)

x = L.SeparableConv2D(256, (2, 2), activation='relu')(x)
x = L.BatchNormalization()(x)
x = L.GlobalMaxPooling2D()(x)

x = L.Dense(256)(x)
x = L.LeakyReLU()(x)
x = L.Dropout(0.5)(x)

x = L.Dense(64, kernel_regularizer=l2(2e-4))(x)
x = L.LeakyReLU()(x)
x = L.Dropout(0.5)(x)

x = L.Dense(26, activation='softmax')(x)

model = Model(entry, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              validation_data=val_generator,
                              epochs=666,
                              callbacks=[model_ckpt, reduce_lr, early_stop],
                              verbose=1)

model.summary()

# Load the trained model
model = load_model('BrailleNet.h5')

# Evaluate the model on the validation generator
eval_results = model.evaluate_generator(val_generator)

# Extract the accuracy from the evaluation results
accuracy = eval_results[1]

# Print the accuracy
print('Validation Accuracy: {:.2%}'.format(accuracy))
