import keras
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Flatten, Dropout, Activation, BatchNormalization, Input
from keras import models
from keras.models import Model
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import random
from preprocessing import get_kmer_from_seq,get_params,DataGenerator_from_seq
from architecture import build_model

#path for the training file
filepath_train="taxonomy_data.txt"
#path for the validating file
filepath_val="taxonomy_data.txt"


#paths for saving model and loss
filepath_loss="Multi_task_model.loss"
filepath_model="best_model.h5"

d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
f_matrix,f_labels=get_kmer_from_seq(filepath_train)
f_matrix_val,f_labels_val=get_kmer_from_seq(filepath_val)
params = get_params()

training_generator = DataGenerator_from_seq(f_matrix, f_labels, **params)
val_generator = DataGenerator_from_seq(f_matrix_val, f_labels_val, **params)

model=build_model()
print(model.summary())
model.compile(optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy'])

CallBacks = [EarlyStopping(monitor='val_loss', patience=5),
                ModelCheckpoint(filepath=filepath_model, monitor='val_loss', save_best_only=True)]

result = model.fit_generator(training_generator,
	epochs=100,
	verbose=1,
	validation_data=val_generator,
	callbacks=CallBacks
	)
with open(filepath_loss,"wb") as f:
	f.write(str(result.history).encode())
