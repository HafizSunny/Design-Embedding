# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:25:17 2020

@author: mhrahman
"""
#%%
import json,os , glob
import re
import pandas as pd
from collections import Counter
import seaborn as sns
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
# Apply Design process model
p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Correct deisgner\FBS_action'
#p_path = r'D:\Molla\Stoughton_data\For_Journal\FBS_action'
#p_path = r'D:\Molla\Uark_Data\Extracted_data\FBS_action'
os.chdir(p_path)
all_file = os.listdir(p_path)

coll = []
for i in range(len(all_file)):
    fi_name = os.path.splitext(all_file[i])[0]
    g = pd.read_csv(all_file[i])
    g.drop(g.index[g['x'] == 'Camera'],inplace = True)
    dct_g = Counter(g['x'])
    dct_g['Student'] = fi_name
    coll.append(dct_g)
    
    
Action_fre = pd.DataFrame(coll)
Action_fre = Action_fre.fillna(0)
Action_fre = Action_fre.set_index("Student")
Designers = list(Action_fre.index)
#%%# Saving the embedding 
loc = r'D:\Molla\Stoughton_data\For_Journal\Saved_embedding'
#loc = r'D:\Molla\Uark_Data\Result\Saved_emd'
os.chdir(loc)
with open('Action_pre.pkl','wb') as f:
    pickle.dump(Action_fre.values,f)

#%%
# Clustering using -----------------
#######################################
reduced_data_PCA = PCA(n_components=3).fit_transform(Action_fre)

### clustering using original data
reduced_data = Action_fre.values

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



kmeans = KMeans(init='k-means++', n_clusters= 3, n_init=10)
kmeans.fit(reduced_data)

labels = kmeans.fit_predict(reduced_data)
print(labels)

fig = plt.figure(figsize=(5.5, 3))
ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
ax.scatter(reduced_data_PCA[:, 1], reduced_data_PCA[:, 0], reduced_data_PCA[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
plt.show()

plt.scatter(reduced_data_PCA[:, 0], reduced_data_PCA[:, 1], c=labels, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)




action_clust = []
for j in range(kmeans.n_clusters):
    at = []
    for i in np.where(kmeans.labels_ == j)[0]:
        at.append(Designers[i])
    action_clust.append(at)
df = pd.DataFrame(action_clust).T 
columns = ["0", "1", "2"]   
df.columns = columns
out_path = r'D:\Molla\Stoughton_data\Distribution'
os.chdir(out_path)
#df.to_csv('Action_fre_cluster.csv', index = False)

#######################################
# Finding_mean and std
os.chdir(out_path)
df = pd.read_csv('cluster.csv')

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
df.to_csv('Action_cluster.csv', index = True) 

#%%
#X_means clustering -------------------------------------------------------------
reduced_data = PCA(n_components=3).fit_transform(Action_fre)
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(reduced_data,amount_initial_centers).initialize()
xmeans_instance = xmeans(reduced_data, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
# Visualize clustering results
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, reduced_data,marker = 'o',markersize= 20)
visualizer.append_cluster(centers, None, marker='*', markersize=100)
visualizer.show()
#%%#For converting clusters assignment
clusts = []
order = np.concatenate(clusters).argsort()
clusts = list(np.concatenate([ [i]*len(e) for i,e in enumerate(clusters) ])[order])
print(clusts)
#%%
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
#out_path = r'D:\Molla\Uark_Data\Result\Action_prefrence'
out_path = r'D:\Molla\Stoughton_data\For_Journal\Result'
os.chdir(out_path)
fig.savefig('Action_fre.tif', format='tif', dpi=300)
#%%# For getting the student ID
action_clust = []
for j in range(len(clusters)):
    at = []
    for i in np.where(np.array(clusts) == j)[0]:
        at.append(Designers[i])
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
#out_path = r'D:\Molla\Stoughton_data\Result from X_means\Action_frequency'
#out_path = r'D:\Molla\Uark_Data\Result\Action_prefrence'
out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\Action_fre'
os.chdir(out_path)
df.to_csv('Action_fre.csv', index = True) 
#%%# Performance analysis
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
#out_path = r'D:\Molla\Stoughton_data\Result from X_means\Action_frequency'
#out_path = r'D:\Molla\Uark_Data\Result\Action_prefrence'
out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\Action_fre'
os.chdir(out_path)
perfor.to_csv('Performance.csv', index = True) 
#%%
# For detail data analysis
#--- After line 347
Cluster_1 = [x for x in df['0'] if x is not None]
Cluster_2 = [x for x in df['1'] if x is not None]
Cluster_3 = [x for x in df['2'] if x is not None]

p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Correct deisgner\FBS_action'
os.chdir(p_path)

def coun(file):
    coll = []
    for i in range(len(file)):
        fi_name = os.path.splitext(file[i])[0]
        g = pd.read_csv(os.path.join(file[i])+'.csv')
        g.drop(g.index[g['x'] == 'Camera'],inplace = True)
        dct_g = Counter(g['x'])
        dct_g['Student'] = fi_name
        coll.append(dct_g)
    return coll
clus1 = coun(Cluster_1)
clus1 = pd.DataFrame(clus1)
clus1 = clus1.set_index("Student")

clus2 = coun(Cluster_2)
clus2 = pd.DataFrame(clus2)
clus2 = clus2.set_index("Student")

clus3 = coun(Cluster_3)
clus3 = pd.DataFrame(clus3)
clus3 = clus3.set_index("Student")

clus3.plot(kind = 'bar')
plt.xticks(rotation=30, horizontalalignment="center")
plt.xlabel("Designer")
plt.ylabel("Frequency of design process stage")

#%% bar plot of action preference
example = {'Formulation':18,'Reformulation 2':7,'Synthesis': 85, 'Analysis':13,'Evaluation':1,'Reformulation 3': 1, 'Reformulation 1':0}
keys = list(example.keys())
vals = [(example[k]) for k in keys]

fig, ax = plt.subplots(figsize = (6,4))
fig = sns.barplot(x=keys, y=vals)
ax.set(xlabel = 'Design process stage', ylabel = 'Frequency')
ax.set_xticklabels(labels = keys, fontsize=10, rotation=15)
out_path = r'D:\Molla\Stoughton_data\Result from X_means\Action_frequency'
os.chdir(out_path)
plt.savefig('Action_dist.tif', format='tif', dpi=300)