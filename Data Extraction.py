# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:03:10 2021

@author: mhrahman
"""
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
path = "D:\Molla\Stoughton_data"
file_list = [x for x in os.listdir(path) if x.endswith('.json')]


for i in range(len(file_list)):
    action = []
    file_name = os.path.splitext(file_list[i])[0]
    with open (os.path.join(path,file_list[i])) as json_file:
        json_text = json.load(json_file)
        for j in range(len(json_text['Activities'])):
            ac = ""
            search_name = bool(re.search("^stoughton*",json_text['Activities'][j]['File']))
            if len(json_text['Activities'][j].keys()) == 4:
                Pro_name = json_text['Activities'][j]['Project']
                if Pro_name == 'Stoughton High School':
                    ac = ''.join(list(set(sorted(json_text["Activities"][j].keys())) - {'File', 'Project', 'Timestamp'}))
            elif search_name == True:
                ac = ''.join(list(set(sorted(json_text["Activities"][j].keys())) - {'File', 'Project', 'Timestamp'}))
            else:
                print("N/A")
            action.append(ac)
            fil_action = [string for string in action if string != ""]
        df = pd.DataFrame(fil_action)
        output_path = r'D:\Molla\Stoughton_data\Data_csv'
        os.chdir(output_path)
        df.to_csv('{}.csv'.format(file_name),index = False, header = 'x')


all_actions = []
csv_list = os.listdir(output_path)
for i in range(len(csv_list)):
    a = list(pd.read_csv(csv_list[i])['0'])
    all_actions.append(a)


#Optimize-it data 
final_path = r'D:\Molla\Stoughton_data\Final Data'
orginal_list = os.listdir(final_path)   

or_list = [x.split(".")[0] for x in orginal_list]
full_list = [x.split(".")[0] for x in file_list]
or_list_set = set(or_list)
intersection = list(or_list_set.intersection(full_list))

Json_list = [str(element) + '.json' for element in intersection]

for i in range(len(Json_list)):
    action = []
    file_name = os.path.splitext(Json_list[i])[0]
    with open (os.path.join(path,Json_list[i])) as json_file:
        json_text = json.load(json_file)
        for j in range(len(json_text['Activities'])):
            ac = ""
            search_name = bool(re.search("^optimize-it*",json_text['Activities'][j]['File']))
            if len(json_text['Activities'][j].keys()) == 4:
                Pro_name = json_text['Activities'][j]['Project']
                if Pro_name == 'Optimize It!':
                    ac = ''.join(list(set(sorted(json_text["Activities"][j].keys())) - {'File', 'Project', 'Timestamp'}))
            elif search_name == True:
                ac = ''.join(list(set(sorted(json_text["Activities"][j].keys())) - {'File', 'Project', 'Timestamp'}))
            else:
                print("N/A")
            action.append(ac)
            fil_action = [string for string in action if string != ""]
        df = pd.DataFrame(fil_action)
        output_path = r'D:\Molla\Stoughton_data\Optimize_it_data'
        os.chdir(output_path)
        df.to_csv('{}.csv'.format(file_name),index = False, header = 'x')



# Annual net energy of Optimize-it
dics = []
for i in range(len(Json_list)):
   act = [] 
   f_name = os.path.splitext(Json_list[i])[0]
   with open(os.path.join(path, Json_list[i])) as Js:
       Ja = json.load(Js)
       for j in range(len(Ja['Activities'])):
           if len(Ja['Activities'][j].keys()) == 4:
               P = Ja['Activities'][j]['Project']
               if P == 'Optimize It!':
                   a = ''.join(list(set(sorted(Ja["Activities"][j].keys())) - {'File', 'Project', 'Timestamp'}))
                   if a == 'PvAnnualAnalysis':
                       b = Ja['Activities'][j]['PvAnnualAnalysis']['Solar']['Total']
                       act.append(b)
               else:
                   print('N/A')
   if len(act) != 0:
       dics.append([f_name,act[-1]])
ANE_df = pd.DataFrame(dics,columns = ['Student','ANE'])
p = r'D:\Molla\Stoughton_data\Distribution'
os.chdir(p)
ANE_df.to_csv('optimize.csv',index = False)        


opt_path = r'D:\Molla\Stoughton_data\Optimize_it_data'
os.chdir(opt_path)
op_files = os.listdir(opt_path)

op_coll = []
fi = []
for i in range(len(op_files)):
    fi_name = os.path.splitext(op_files[i])[0]
    g = pd.read_csv(op_files[i])
    g.drop(g.index[g['0'] == 'Camera'],inplace = True)
#    dct_g = Counter(g['0'])
#    dct_g['Student'] = fi_name
#    op_coll.append(dct_g)
    c = len(g['0'])
    op_coll.append(c)
    fi.append(fi_name)

d = {'Studnet':fi,'Op_action':op_coll }
op_df = pd.DataFrame(d)
op_df = op_df.set_index('Studnet')
        
# Read_Final_CSV
path = r'D:\Molla\Stoughton_data\Distribution'
os.chdir(path)
f_list = os.listdir(path) 
final = pd.read_csv('Final.csv')
op_ANE = pd.read_csv('optimize.csv')
final_f_op = final.loc[final['Computer ID'].isin(fi)]   
final_f_op = final_f_op.set_index('Computer ID')
final_f_op = final_f_op.loc[fi]

op_ANE = op_ANE.set_index('Student')
op_ANE = op_ANE.loc[fi]         
            
Merged = ANE_df.combine_first(final)            
Final_merged = Merged.dropna()
Final_merged = Final_merged.rename(columns = {'ANE':'Optimize_ANE'})
Final_merged.to_csv('Final_merged.csv',index = False)

op_final = pd.concat([op_ANE,op_df],axis = 1)

Total_final = pd.concat([op_final,final_f_op],axis = 1)
Total_final = Total_final.drop(['Total cost','Ratio','Action len'],axis = 1)
Total_final = Total_final.rename(columns = {'Wo_camera':'St_action'})

# Count each of the student's design
path = r'D:\Molla\Stoughton_data\Final Data'
os.chdir(path)
f_list = os.listdir(path)
g = pd.read_csv(f_list[1])
g.drop(g.index[g['0'] == 'Camera'],inplace=True)    

ba = pd.DataFrame.from_dict(Counter(g['0']), orient='index')
sns.heatmap(Total_final.corr(),annot = True, vmin=-1,vmax=1,center=0)


coll = []
for i in range(len(f_list)):
    fi_name = os.path.splitext(f_list[i])[0]
    g = pd.read_csv(f_list[i])
    g.drop(g.index[g['0'] == 'Camera'],inplace = True)
    dct_g = Counter(g['0'])
    dct_g['Student'] = fi_name
    coll.append(dct_g)

#%%

#%%

path = "D:\Molla\Stoughton_data"
os.chdir(path)
file_list = [x for x in os.listdir(path) if x.endswith('.json')]

for i in range(len(file_list)):
    os.chdir(path)
    action = []
    timestamp = []
#    Val = []
    file_name = os.path.splitext(file_list[i])[0]
    with open (os.path.join(path,file_list[i])) as json_file:
        json_text = json.load(json_file)
        for j in range(len(json_text['Activities'])):
            ac = ""
            time = ""
            value = ""
            search_name = bool(re.search("^stoughton*",json_text['Activities'][j]['File']))
            if len(json_text['Activities'][j].keys()) == 4:
                Pro_name = json_text['Activities'][j]['Project']
                if Pro_name == 'Stoughton High School':
                    ac = ''.join(list(set(sorted(json_text["Activities"][j].keys())) - {'File', 'Project', 'Timestamp'}))
                    if ac == 'Graph Tab':
                        if json_text['Activities'][j]['Graph Tab'] == 'Energy':
                            ac = 'Graph Tab Energy'
                        elif json_text['Activities'][j]['Graph Tab'] == 'Cost':
                            ac = 'Graph Tab Cost'
                        else:
                            ac = 'Graph Tab'
                    time = json_text["Activities"][j]['Timestamp']
#                    if type(json_text["Activities"][j][ac]) == dict:
#                        if 'New Value'in (json_text['Activities'])[j][ac].keys():
#                            value = json_text["Activities"][j][ac]['New Value']
#                        else:
#                            value = "None"
#                    else:
#                        value = "None"
            elif search_name == True:
                ac = ''.join(list(set(sorted(json_text["Activities"][j].keys())) - {'File', 'Project', 'Timestamp'}))
            else:
                print("N/A")
            action.append(ac)
            timestamp.append(time)
#            Val.append(value)            
            fil_action = [string for string in action if string != ""]
            fil_time = [string for string in timestamp if string != ""]
#            fil_Val = [string for string in Val if string != ""]
            df = pd.DataFrame(list(zip(fil_time,fil_action)), columns = ['Timestamp' , 'Action'])
            output_path = r'D:\Molla\Stoughton_data\Data_stoughtn'
            os.chdir(output_path)
            df.to_csv('{}.csv'.format(file_name),index = False)

des_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
main_file  = os.listdir(r'D:\Molla\Stoughton_data\Final Data')

file = os.listdir(output_path)
os.chdir(output_path)

for f in file:
    if f in main_file:
        shutil.copy(f, des_path)

removal_list = ['Undo','Edit Human','Save','Top View','Change Time','Spin View','Change Date',
                'Change City', 'Change Time and Date', 'Camera']

ori_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Correct deisgner'
Action_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Correct deisgner\Only_action'

for i in range(len(main_file)):
    os.chdir(ori_path)
    file_name = os.path.splitext(main_file[i])[0]
    action = list(pd.read_csv(main_file[i])['Action'])
    l3 = [x for x in action if x not in removal_list]
    ac_df = pd.DataFrame(l3)
    os.chdir(Action_path)
    ac_df.to_csv('{}.csv'.format(file_name),index = False, header = None)

#Unique actions extraction
    
p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Correct deisgner\Only_action'
os.chdir(p_path)
all_file = os.listdir(p_path)

all_action = []
for i in range(len(all_file)):
    p = list(pd.read_csv(all_file[i],header = None)[0])
    all_action.append(p)
Merge_action = list(itertools.chain(*all_action))
Unique_actions = set(Merge_action)
process = list(Unique_actions)

Unique_Df = pd.DataFrame(list(Unique_actions))
Unique_Df.to_csv('Unique.csv',index = False,header = None)

############# Time gap Analysis ########################----------------------
#%%
path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
os.chdir(path)
f_list = os.listdir(path)


for i in range(len(f_list)):
    os.chdir(path)
    csv = pd.read_csv(f_list[i])
    csv.drop(csv.index[csv['Action'] == 'Camera'], inplace = True)
    csv = csv.reset_index(drop = True)
    ba = []
    for k in range(1,len(csv.Timestamp)):
        gap = (pd.to_datetime(csv.Timestamp[k]) - pd.to_datetime(csv.Timestamp[k-1])).total_seconds()
        if gap > 300:
            gap = 0
        ba.append(gap)
    ba.insert(0,0)
    csv.insert(3,"Timegap",ba, True)        
    file_name = os.path.splitext(f_list[i])[0]
    p_path = r'D:\Molla\Stoughton_data\Data_stoughtn\Data_value\Final_Data_value'
    os.chdir(p_path)
    csv.to_csv('{}.csv'.format(file_name),index = False)

Only_action_path = r'D:\Molla\Stoughton_data\Detail_data_WO_camera\Only Action'

for i in range(len(f_list)):
    os.chdir(final_path)
    file_name = os.path.splitext(f_list[i])[0]
    action = list(pd.read_csv(f_list[i])['0'])
    l3 = [x for x in action if x not in removal_list]
    ac_df = pd.DataFrame(l3)
    os.chdir(Only_action_path)
    ac_df.to_csv('{}.csv'.format(file_name),index = False, header = None)
    
