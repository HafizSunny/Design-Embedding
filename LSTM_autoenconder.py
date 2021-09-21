# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:01:28 2020

@author: mhrahman
"""
#%%
from keras.preprocessing import text
import numpy as np
import pandas as pd
import os
import csv
#nltk
import nltk
from nltk import bigrams
from nltk import word_tokenize,sent_tokenize
#keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Embedding, Input,LSTM
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers.core import RepeatVector
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans,DBSCAN
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import itertools
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
#%%

#Data Prep
p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
#p_path = r'D:\Molla\Uark_Data\Extracted_data\Valid_action'
os.chdir(p_path)
all_file = os.listdir(p_path)

seq = []
Designer = []
for i in range(len(all_file)):
    fi_name = os.path.splitext(all_file[i])[0]
    csv = pd.read_csv(all_file[i])
    csv = list(csv.Action)
    seq.append(csv)
    Designer.append(fi_name)

sal = []
for j in range(len(seq)):
    bal = []
    for i in range(len(seq[j])):
        a = seq[j][i].replace(" ","")
        bal.append(a)
    sal.append(bal)
    
final_sequence = []
for i in range(len(seq)):
    a = " ".join(sal[i])
    final_sequence.append(a)




num_words = 200
maxlen = 700
embed_dim = 100
batch_size = 16


tokenizer = Tokenizer(num_words = num_words, split=' ')
tokenizer.fit_on_texts(final_sequence)
seqs = tokenizer.texts_to_sequences(final_sequence)
pad_seqs = pad_sequences(seqs, maxlen,padding= 'post',truncating = 'post')
No_sequence = len(final_sequence)
#%%
encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Word-Embedding', mask_zero=False)
x = emb_layer(encoder_inputs)
state_h = Bidirectional(LSTM(128, activation='relu',name='Encoder-Last-LSTM'))(x)
#state_h_1 = LSTM(64,activation = 'relu',return_sequences = False)(state_h)
encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)

decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
decoder_lstm = Bidirectional(LSTM(128, return_sequences=True, name='Decoder-LSTM-before'))
decoder_lstm_output = decoder_lstm(decoded)
decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')
decoder_outputs = decoder_dense(decoder_lstm_output)

seq2seq_Model = Model(encoder_inputs, decoder_outputs)
seq2seq_Model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss='sparse_categorical_crossentropy')
history = seq2seq_Model.fit(pad_seqs, np.expand_dims(pad_seqs, -1),
          batch_size=batch_size,
          epochs=100)

print(seq2seq_Model.summary())
path = r'D:\Molla\Stoughton_data\For_Journal\Result\LSTM_autoencoder'
#path = r'D:\Molla\Uark_Data\Result\LSTM_autoencoder'
os.chdir(path)
plot_model(seq2seq_Model,show_shapes = True ,to_file='model.png')
sentence_vec = encoder_model.predict(pad_seqs)
seq2seq_Model.save('my_model.h5')
#%%
#save Embedding
import pickle
loc = r'D:\Molla\Stoughton_data\For_Journal\Saved_embedding'
#loc = r'D:\Molla\Uark_Data\Result\Saved_emd'
os.chdir(loc)
with open('LSTM.pkl','wb') as f:
    pickle.dump(sentence_vec,f)    
#%%
def elbow_plot(matrix):
    wcss = []
    for i in range(1,10):
        Kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state= 42)
        Kmeans.fit(matrix)
        wcss.append(Kmeans.inertia_)
    plt.plot(range(1,10), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

elbow_plot(sentence_vec)
#%%
def plot_kmean(num_cluster,vector,n_component):
    reduced_data_PCA = PCA(n_components= n_component).fit_transform(vector)
    kmeans = KMeans(init='k-means++', n_clusters= num_cluster, n_init=10)
    kmeans.fit(vector)
    labels = kmeans.fit_predict(vector)
    print(labels)    
    fig = plt.figure(figsize=(5.5, 3))
    ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
    ax.scatter(reduced_data_PCA[:, 1], reduced_data_PCA[:, 0], reduced_data_PCA[:, 2],
              c=labels.astype(np.float), edgecolor="k", s=50)
    plt.show()
    return kmeans
    
kmeans = plot_kmean(4,sentence_vec,3)

action_clust = []
for j in range(kmeans.n_clusters):
    at = []
    for i in np.where(kmeans.labels_ == j)[0]:
        at.append(Designer[i])
    action_clust.append(at)
df = pd.DataFrame(action_clust).T 
columns = ["0", "1","2","3"]   
df.columns = columns

## LOAD design_output
path = r'D:\Molla\Stoughton_data\Distribution'
os.chdir(path)
design_output = pd.read_csv('Design_output.csv')
design_output.set_index('Computer ID')

mean = []
std = []
for i in range(len(df.columns)):
    cluster_wise = []
    for j in range(len(df['{}'.format(i)])):
        design = df['{}'.format(i)][j]
        if design in list(design_output['Computer ID']):
            a = design_output.loc[design_output['Computer ID'] == design, 'Co-efficient'].iloc[0]
            cluster_wise.append(a)
            m = np.mean(cluster_wise)
            s = np.std(cluster_wise)
    mean.append(m)
    std.append(s)
            
df.loc[len(df)] = mean
df.loc[len(df)] = std
df = df.rename(index = {df.index[-2]:'mean',df.index[-1]:'std'})   
out_path = r'D:\Molla\Stoughton_data\Distribution'
os.chdir(out_path)
df.to_csv('_cluster.csv', index = True) 

dbscan_opt=DBSCAN(eps=5,min_samples=10)
dbscan_opt.fit(sentence_vec)

tsne = TSNE(n_components=2,perplexity=5).fit_transform(sentence_vec)

#%%
#X_means clustering -------------------------------------------------------------
reduced_data = PCA(n_components=3).fit_transform(sentence_vec)
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(reduced_data,amount_initial_centers).initialize()
xmeans_instance = xmeans(reduced_data, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
# Visualize clustering results
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, reduced_data, marker = 'o',markersize=20)
visualizer.append_cluster(centers, None, marker='*', markersize=100)
visualizer.show()
#%%
#For converting clusters assignment
clusts = []
order = np.concatenate(clusters).argsort()
clusts = list(np.concatenate([ [i]*len(e) for i,e in enumerate(clusters) ])[order])
print(clusts)

#plot cluster 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(np.array(centers)[:, 1],
            np.array(centers)[:, 0],
            np.array(centers)[:, 2],
            s = 250,
            marker='o',
            c='red',
            label='centroids')
scatter = ax.scatter(reduced_data[:, 1], reduced_data[:, 0], reduced_data[:, 2],
                     c=clusts,s=20, cmap='winter')


#ax.set_title('X-Means Clustering')
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.set_zlabel('Principal component 3')
ax.legend()
plt.show()
out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\LSTM_autoencoder'
#out_path = r'D:\Molla\Uark_Data\Result\LSTM_autoencoder'
os.chdir(out_path)
fig.savefig('LSTM_auto.tif', format='tif', dpi=300)
#%%
# For getting the student ID
action_clust = []
for j in range(len(clusters)):
    at = []
    for i in np.where(np.array(clusts) == j)[0]:
        at.append(Designer[i])
    action_clust.append(at)
df = pd.DataFrame(action_clust).T
columns = []
for i in range(len(clusters)):
    columns.append(i)
columns = list(map(str,columns))  
df.columns = columns
#%%
## LOAD design_output
path = r'D:\Molla\Stoughton_data\Distribution'
#path = r'D:\Molla\Uark_Data\Design_out'
os.chdir(path)
design_output = pd.read_csv('Design_output.csv')
design_output['Computer ID'] = design_output['Computer ID'].astype('str')
design_output.set_index('Computer ID')
#%%
mean = []
std = []
for i in range(len(df.columns)):
    cluster_wise = []
    for j in range(len(df['{}'.format(i)])):
        design = df['{}'.format(i)][j]
        if design in list(design_output['Computer ID']):
            a = design_output.loc[design_output['Computer ID'] == design, 'Co-efficient'].iloc[0]
            cluster_wise.append(a)
            m = np.mean(cluster_wise)
            s = np.std(cluster_wise)
    mean.append(m)
    std.append(s)
            
df.loc[len(df)] = mean
df.loc[len(df)] = std
df = df.rename(index = {df.index[-2]:'mean',df.index[-1]:'std'})   
out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\LSTM_autoencoder'
#out_path = r'D:\Molla\Uark_Data\Result\LSTM_autoencoder'
os.chdir(out_path)
df.to_csv('LSTM_autoencoder.csv', index = True) 
#%%
# Performance analysis

performance = []
for i in range(len(df.columns)):
    cluster_wise = []
    for j in range(len(df['{}'.format(i)])):
        design = df['{}'.format(i)][j]
        if design in list(design_output['Computer ID']):
            a = design_output.loc[design_output['Computer ID'] == design, 'Co-efficient'].iloc[0]
            cluster_wise.append(a)
    performance.append(cluster_wise)
perfor = pd.DataFrame(performance).T
out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\LSTM_autoencoder'
#out_path = r'D:\Molla\Uark_Data\Result\LSTM_autoencoder'
os.chdir(out_path)
perfor.to_csv('Performance.csv', index = True) 