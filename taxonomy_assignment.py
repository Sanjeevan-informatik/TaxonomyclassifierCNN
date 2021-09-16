import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#dictionary for one-hot encoding
d_nucl={"A":0,"C":1,"G":2,"T":3,"N":4}
#get seq, labels and locations for 50-mer
#default format for each line of training files: seq+"\t"+label+"\t"+location
def get_taxonomy_seq(filepath):
  f=open(filepath,"r").readlines()
  f_matrix=[]
  f_labels=[]

  for i in f:
    i=i.strip().split("\t")
    f_matrix.append(i[0])

  return f_matrix

def data_generation(matrix):
    x_train=[]
    index= len(matrix)
    for i in range(index):
        seq=matrix[i]
        seq_list=[j for j in seq]
        x_train.append(seq_list)
    x_train=np.array(x_train)
    x_tensor=np.zeros(list(x_train.shape)+[5])
    for row in range(len(x_train)):
        for col in range(50):
            x_tensor[row,col,d_nucl[x_train[row,col]]]=1
    return x_tensor


def lablel_taxonomy_data():
    
    filepath ="sequence_data.txt"
    data =get_taxonomy_seq(filepath)
    val_generator = data_generation(data)
    val_generator = val_generator.reshape(val_generator.shape[0], (val_generator.shape[1]*val_generator.shape[2]))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(val_generator)

    x1_limit = [-1.5,0.0,1.0]
    x2_limit = [-1.5,0.0,1.0]
    length = len(X_pca[:,0])
    a = [None] * length

    for i in range(length):
        x= -3
        taxonamy_value =-1
        taxonamy_assign = False 

        for x1_limit_data in range(len(x1_limit)):
            x=x+3
            for x2_limit_data in range(len(x2_limit)): 
                if X_pca[i, 0] < x1_limit[x1_limit_data] and X_pca[i,1] < x2_limit[x2_limit_data]:
                    a[i] =x2_limit_data

                    taxonamy_assign = True 
                    break

        if taxonamy_assign == False:
            a[i] =3

        with open("taxonomy_data.txt", 'w') as f:
            for i in range(1000):
                seq = data[i]+"\t"+str(a[i])+"\n"
                f.write(seq)
            f.close()