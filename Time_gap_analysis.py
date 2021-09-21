# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 00:20:45 2020

@author: mhrahman
"""
#%%
import json,os , glob, shutil
import re
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.stats as st
from sklearn import preprocessing
import pickle

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


#%%    
## convert to time gap in second------------------------
#p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
p_path = r'D:\Molla\Uark_Data\Extracted_data\Valid_action'
os.chdir(p_path)
all_file = os.listdir(p_path)
#%%
csv = pd.read_csv(all_file[25])
csv = csv[csv.Timegap != 0]
ax = plt.plot()
for j, txt in enumerate(list(csv.Action)):
    ax.annotate()
            
plt.plot(pd.to_datetime(pd.read_csv(f_list[0]).Timestamp))

plt.plot(csv.Timegap)
plt.ylabel("Time gaps in second")

def Greaterthannumber(val,actions,number):
    if len(val) != len(actions):
        return
    for i in range(0,len(actions)):
        if val[i] > number:
            plt.annotate(actions[i], (i,val[i]),rotation = -90, fontsize = 8)

Greaterthannumber(csv.Timegap,csv.Action,20)
plt.show()

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110,120,130,140,150,160,170,180,190,200]
fu = csv['Timegap'].value_counts(bins=bins, sort=False)
bins = list(range(1, int(max(csv.Timegap)) ,1))

#%%
# Frequency
def pdf (file_list):
    for i in range(len(file_list)):
        os.chdir(p_path)
        file_nm = os.path.splitext(file_list[i])[0]
        csv = pd.read_csv(file_list[i])
        csv = csv[csv.Timegap != 0]
        bins = list(range(1, int(max(csv.Timegap)) ,1))
        sns.histplot(csv.Timegap,bins = bins)
        #out = r'D:\Molla\Stoughton_data\Distribution\PDF_timegap'
        out = r'D:\Molla\Uark_Data\Result\Timegap\PDF'
        os.chdir(out)
        plt.savefig('{}.png'.format(file_nm),bbox_inches='tight',dpi = 600)
        plt.close()

pdf(all_file)

def cdf (file_list):
    for i in range(len(file_list)):
        os.chdir(p_path)
        file_nm = os.path.splitext(file_list[i])[0]
        csv = pd.read_csv(file_list[i])
        csv = csv[csv.Timegap != 0]
        sns.kdeplot(csv.Timegap,cumulative = True)
        #out = r'D:\Molla\Stoughton_data\Distribution\CDF_timegap'
        out = r'D:\Molla\Uark_Data\Result\Timegap\CDF'
        os.chdir(out)
        plt.savefig('{}.png'.format(file_nm),bbox_inches='tight',dpi = 600)
        plt.close()

cdf(all_file)
#%%

def get_best_distribution(data):
#    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min","expon","pareto", "genextreme","gamma","beta",'halfcauchy','lognorm']
    dist_names = ["genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))
    return best_dist, best_p, params[best_dist]
#%%

def pdf (file_list):
    for i in range(len(file_list)):
        os.chdir(p_path)
        file_nm = os.path.splitext(file_list[i])[0]
        csv = pd.read_csv(file_list[i])
        csv = csv[csv.Timegap != 0]
        bins = list(range(1, int(max(csv.Timegap)) ,1))
        sns.distplot(csv.Timegap,bins = bins)


y = np.asarray(list(csv.Timegap))
x = np.arange(len(y))
number_of_bins = len(y)
bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)
h = plt.hist(y, bins = bin_cutoffs, color='red')
pdf_fitted = dist.pdf(np.arange(len(y)),param[:-2],loc = param[-2],scale = param[-1])
scale_pdf = np.trapz(h[0],h[1][:-1])/np.trapz(pdf_fitted,x)
pdf_fitted *= scale_pdf 
plt.plot(pdf_fitted) 
plt.show()  

#%% 
def pdf_fitted(csv):
    y = np.asarray(list(csv.Timegap))
    x = np.arange(len(y))
    number_of_bins = len(y)
#    bin_cutoff = np.linspace(np.percentile(y,0),np.percentile(y,99),number_of_bins)
    h = plt.hist(y,bins= 300)
    k = get_best_distribution(y)
    dist = getattr(st,k[0])
    param = k[2]
#    pdf_fit = dist.pdf(x,param[:-2],loc = param[-2],scale = param[-1])
    pdf_fit = dist.pdf(x,param[0],param[1])
    scaled_pdf = np.trapz(h[0],h[1][:-1])/np.trapz(pdf_fit,x)
#    plt.xlim(0,300)
    pdf_fit *= scaled_pdf
    plt.plot(pdf_fit,'--g',linewidth = 0.6,label = 'GEV distribution')
    plt.legend(loc = 'upper right')
#    plt.xlabel("Time gap (second)")
#    plt.ylabel("Frequecy of time gap")
    plt.show()    
#%%    
#p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
p_path = r'D:\Molla\Uark_Data\Extracted_data\Valid_action'
os.chdir(p_path)
all_file = os.listdir(p_path)
for i in range(len(all_file)):
    os.chdir(p_path)
    file_nm = os.path.splitext(all_file[i])[0]
    csv = pd.read_csv(all_file[i])
    csv = csv[csv.Timegap != 0]
    pdf_fitted(csv)
    #out = r'D:\Molla\Stoughton_data\Distribution\New_dist'
    out = r'D:\Molla\Uark_Data\Result\Timegap\Fitter_dist'
    os.chdir(out)
    plt.savefig('{}.png'.format(file_nm),bbox_inches='tight',dpi = 600)
    plt.close()

# Distribution for all
#%%  
#p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
p_path = r'D:\Molla\Uark_Data\Extracted_data\Valid_action'
os.chdir(p_path)
all_file = os.listdir(p_path)


file = []
dist_name = []
parameters = []
param_1 = []
param_2 = []
param_3 = []

for i in range(len(all_file)):
    os.chdir(p_path)
    file_nm = os.path.splitext(all_file[i])[0]
    csv = pd.read_csv(all_file[i])
    csv = csv[csv.Timegap != 0]
    k = get_best_distribution(csv.Timegap)
    dist_name.append(k[0])
    file.append(file_nm)
    a = k[2][0]
    b = k[2][1]
    c = k[2][2]
    param_1.append(a)
    param_2.append(b)
    param_3.append(c)

Df = pd.DataFrame({
                   'Param 1': param_1,
                   'param 2':param_2,
                   'param 3': param_3})    

Only_values = Df.values
#%%# Saving the embedding 
#loc = r'D:\Molla\Stoughton_data\Models\New folder\Saved_embedding'
loc = r'D:\Molla\Uark_Data\Result\Saved_emd'
#loc = r'D:\Molla\Stoughton_data\For_Journal\Saved_embedding'
os.chdir(loc)
with open('Timegap.pkl','wb') as f:
    pickle.dump(Df.values,f)
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

elbow_plot(Only_values)

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
    
kmeans = plot_kmean(3,Only_values,3)

action_clust = []
for j in range(kmeans.n_clusters):
    at = []
    for i in np.where(kmeans.labels_ == j)[0]:
        at.append(file[i])
    action_clust.append(at)
df = pd.DataFrame(action_clust).T 
columns = ["0", "1","2"]   
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
df.to_csv('Timegap_cluster.csv', index = True) 




# Additional for distribution-----
distribution = "expon"
data = np.asarray(list(csv.Timegap))

dist = getattr(st, distribution)
param = dist.fit(data)

# Get random numbers from distribution
norm = dist.rvs(loc=param[-2], scale=param[-1],size = len(data))
norm.sort()

# Create figure
fig = plt.figure(figsize=(8,5)) 

# qq plot
ax1 = fig.add_subplot(121) # Grid of 2x2, this is suplot 1
ax1.plot(norm,data,"o")
min_value = np.floor(min(min(norm),min(data)))
max_value = np.ceil(max(max(norm),max(data)))
ax1.plot([min_value,max_value],[min_value,max_value],'r--')
ax1.set_xlim(min_value,max_value)
ax1.set_xlabel('Theoretical quantiles')
ax1.set_ylabel('Observed quantiles')
title = 'qq plot for ' + distribution +' distribution'
ax1.set_title(title)

# pp plot
ax2 = fig.add_subplot(122)

# Calculate cumulative distributions
bins = np.percentile(norm,range(0,101))
data_counts, bins = np.histogram(data,bins)
norm_counts, bins = np.histogram(norm,bins)
cum_data = np.cumsum(data_counts)
cum_norm = np.cumsum(norm_counts)
cum_data = cum_data / max(cum_data)
cum_norm = cum_norm / max(cum_norm)

# plot
ax2.plot(cum_norm,cum_data,"o")
min_value = np.floor(min(min(cum_norm),min(cum_data)))
max_value = np.ceil(max(max(cum_norm),max(cum_data)))
ax2.plot([min_value,max_value],[min_value,max_value],'r--')
ax2.set_xlim(min_value,max_value)
ax2.set_xlabel('Theoretical cumulative distribution')
ax2.set_ylabel('Observed cumulative distribution')
title = 'pp plot for ' + distribution +' distribution'
ax2.set_title(title)

# Display plot    
plt.tight_layout(pad=4)
plt.show()

#%%
#X_means clustering -------------------------------------------------------------
reduced_data = PCA(n_components=3).fit_transform(Only_values)
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(reduced_data,amount_initial_centers).initialize()
xmeans_instance = xmeans(reduced_data, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
# Visualize clustering results
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, reduced_data,marker = 'o',markersize = 20)
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
out_path = r'D:\Molla\Uark_Data\Result\Timegap\Result'
#out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\Time_gap'
os.chdir(out_path)
fig.savefig('Timegap.tif', format='tif', dpi=300)
#%%
# For getting the student ID
action_clust = []
for j in range(len(clusters)):
    at = []
    for i in np.where(np.array(clusts) == j)[0]:
        at.append(file[i])
    action_clust.append(at)
df = pd.DataFrame(action_clust).T
columns = []
for i in range(len(clusters)):
    columns.append(i)
columns = list(map(str,columns))  
df.columns = columns
#%%
## LOAD design_output
#path = r'D:\Molla\Stoughton_data\Distribution'
path = r'D:\Molla\Uark_Data\Design_out'
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
#out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\Time_gap'
out_path = r'D:\Molla\Uark_Data\Result\Timegap\Result'
os.chdir(out_path)
df.to_csv('Timegap_cluster.csv', index = True) 
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
#out_path = r'D:\Molla\Stoughton_data\For_Journal\Result\Time_gap'
out_path = r'D:\Molla\Uark_Data\Result\Timegap\Result'
os.chdir(out_path)
perfor.to_csv('Performance.csv', index = True) 
#%%#### FOR PLOT
def pdf_fitted(csv):
    y = np.asarray(list(csv.Timegap))
    # x = np.arange(len(y))
    # number_of_bins = len(y)
    bins = list(range(1, int(max(csv.Timegap)) ,1))
    ax = sns.histplot(csv.Timegap,bins = bins,stat = 'density')
#    k = get_best_distribution(y)
    dist = getattr(st,'genextreme')
    params = dist.fit(y)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    x_min, x_max = ax.get_xlim()
    xs = np.linspace(x_min, x_max, 200)
    ax.plot(xs, dist.pdf(xs, arg, loc=loc, scale=scale), color='r', ls=':', linewidth = 0.5, label='fitted GEV')
    ax.set_xlim(x_min, x_max) 
    # if arg:
    #     pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)* 200
    # else:
    #     pdf_fitted = dist.pdf(x, loc=loc, scale=loc)* 200
#    plt.plot(pdf_fitted, '--g',linewidth = 0.6,label = 'GEV distribution')
    plt.legend(loc = 'upper right')
    plt.show()
    
#p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
p_path = r'D:\Molla\Uark_Data\Extracted_data\Valid_action'
os.chdir(p_path)
all_file = os.listdir(p_path)
for i in range(len(all_file)):
    os.chdir(p_path)
    file_nm = os.path.splitext(all_file[i])[0]
    csv = pd.read_csv(all_file[i])
    csv = csv[csv.Timegap != 0]
    pdf_fitted(csv)
#    out = r'D:\Molla\Stoughton_data\Distribution\PDF_try'
    out = r'D:\Molla\Uark_Data\Result\Timegap\Fitted'
    os.chdir(out)
    plt.savefig('{}.png'.format(file_nm),bbox_inches='tight',dpi = 600)
    plt.close()

#%%
def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

