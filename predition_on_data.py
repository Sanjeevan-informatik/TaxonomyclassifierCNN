import keras
from keras import models
from keras.models import load_model
import random

#path for testing file
filepath_test="taxonomy_data.txt"
#path for trained model
filepath_model="best_model.h5"


d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
f_matrix,y_true=get_kmer_from_seq(filepath_train)
testing_generator = DataGenerator_from_seq_testing(f_matrix)

model=load_model(filepath_model)
hist = model.predict_generator(testing_generator,
	verbose=1
	)
y_pred=[str(i.argmax(axis=-1)) for i in hist]