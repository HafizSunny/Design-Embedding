# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:32:21 2020

@author: mhrahman
"""
#%%
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
#%%
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
    
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(final_sequence)]

# Model training
model = Doc2Vec(tagged_data, size = 100, window = 5, min_count = 1, worker = 7, dm = 0,alpha = 0.025, min_alpha = 0.01 )
model.train(tagged_data, total_examples=model.corpus_count, epochs= 100, start_alpha=0.002, end_alpha=-0.016)
embedding = model.docvecs.doctag_syn0

# Saving the embedding 
#loc = r'D:\Molla\Stoughton_data\Models\New folder\Saved_embedding'
#loc = r'D:\Molla\Uark_Data\Result\Saved_emd'
loc = r'D:\Molla\Stoughton_data\For_Journal\Saved_embedding'
os.chdir(loc)
with open('Doc2vec.pkl','wb') as f:
    pickle.dump(embedding,f)
    
#%%
# Clustering using -----------------
#######################################
reduced_data_PCA = PCA(n_components=3).fit_transform(embedding)

### clustering using original data
reduced_data = embedding

wcss = []
for i in range(1,10):
    Kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state= 42)
    Kmeans.fit(reduced_data)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,10), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show() 



kmeans = KMeans(init='k-means++', n_clusters= 4, n_init=10)
kmeans.fit(reduced_data)

labels = kmeans.fit_predict(reduced_data)
print(labels)

fig = plt.figure(figsize=(5.5, 3))
ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
ax.scatter(reduced_data_PCA[:, 1], reduced_data_PCA[:, 0], reduced_data_PCA[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
plt.show()

# 2D plot
plt.scatter(reduced_data_PCA[:, 0], reduced_data_PCA[:, 1], c=labels, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)



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
df.to_csv('Embedding_cluster.csv', index = True)        

#%%        
#X_means clustering -------------------------------------------------------------
reduced_data = PCA(n_components=3).fit_transform(embedding)
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(reduced_data,amount_initial_centers).initialize()
xmeans_instance = xmeans(reduced_data, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
# Visualize clustering results
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, reduced_data,marker = 'o', markersize= 20)
visualizer.append_cluster(centers, None, marker='*', markersize= 100)
visualizer.show()
out_path = r'D:\Molla\Stoughton_data\Result from X_means\Doc2Vec'
os.chdir(out_path)
fig.savefig('Doc2Vec_main.tif', format='tif', dpi=300)
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
out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\Doc2Vec'
#out_path = r'D:\Molla\Uark_Data\Result\Doc2vec'
os.chdir(out_path)
fig.savefig('Doc2Vec.tif', format='tif', dpi=300)
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
#path = r'D:\Molla\Uark_Data\Design_out'
path = r'D:\Molla\Stoughton_data\Distribution'
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
#out_path = r'D:\Molla\Uark_Data\Result\Doc2vec'
out_path = r'D:\Molla\Uark_Data\Result\Doc2vec'
os.chdir(out_path)
df.to_csv('Doc2Vec.csv', index = True) 
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
out_path = r'D:\Molla\Uark_Data\Result\Doc2vec'
os.chdir(out_path)
perfor.to_csv('Performance.csv', index = True)         
    