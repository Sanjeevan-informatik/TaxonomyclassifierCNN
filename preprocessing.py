import keras
import numpy as np

#dictionary for one-hot encoding
d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}


#set default params for generating batches of 50-mer
def get_params():
	params = {'batch_size': 100,
	'n_classes': 4,
	'shuffle': True}
	return params



#get seq, labels and locations for 50-mer
#default format for each line of training files: seq+"\t"+label+"\t"+location
def get_kmer_from_seq(filepath):
	f=open(filepath,"r").readlines()
	f_matrix=[]
	f_labels=[]

	for i in f:
		i=i.strip().split("\t")
		f_matrix.append(i[0])
		f_labels.append(i[1])
	return f_matrix,f_labels


#get seq  from original dataset
#default format for each line of training files
def get_seq_from_realdata(filepath):
	f=open(filepath,"r").readlines()
	lines=[]
	for i in range(0,len(f),4):
		lines.append(f[i+1].strip())
	f_matrix=[]
	f_index=[]
	sum_loc=0
	for line in lines:
		line=line.strip()
		length_of_read=len(line)
		if length_of_read>=50:
			for i in range(len(line)-49):
				kmer=line[i:i+50]
				f_matrix.append(kmer)
				sum_loc+=1
			f_index.append(sum_loc)
	return f_matrix,f_index


#data generator for generating batches of data from seq
class DataGenerator_from_seq(keras.utils.Sequence):
	def __init__(self, f_matrix, f_labels, batch_size=1024,n_classes=10, shuffle=True):
		self.batch_size = batch_size
		self.labels = f_labels
		self.matrix = f_matrix
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()
	def __len__(self):
		return int(np.ceil(len(self.labels) / self.batch_size))
	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		X, y= self.__data_generation(indexes)
		return X,y
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.labels))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
	def __data_generation(self, index):
		x_train=[]
		for i in index:
			seq=self.matrix[i]
			seq_list=[j for j in seq]
			x_train.append(seq_list)
		x_train=np.array(x_train)
		x_tensor=np.zeros(list(x_train.shape)+[5])
		for row in range(len(x_train)):
			for col in range(50):
				x_tensor[row,col,d_nucl[x_train[row,col]]]=1

		y_label=[self.labels[i] for i in index]
		y_label=np.array(y_label)
		y_label=keras.utils.to_categorical(y_label, num_classes=self.n_classes)
	
		return x_tensor, y_label

#data generator for generating batches of data from 50-mers for testing
class DataGenerator_from_seq_testing(keras.utils.Sequence):
	def __init__(self, f_matrix, batch_size=1024,shuffle=False):
		self.batch_size = batch_size
		self.matrix = f_matrix
		self.shuffle = shuffle
		self.on_epoch_end()
	def __len__(self):
		return int(np.ceil(len(self.matrix) / self.batch_size))
	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		X = self.__data_generation(indexes)
		return X
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.matrix))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
	def __data_generation(self, index):
		x_train=[]
		for i in index:
			seq=self.matrix[i]
			seq_list=[j for j in seq]
			x_train.append(seq_list)
		x_train=np.array(x_train)
		x_tensor=np.zeros(list(x_train.shape)+[5])
		for row in range(len(x_train)):
			for col in range(50):
				x_tensor[row,col,d_nucl[x_train[row,col]]]=1
		return x_tensor