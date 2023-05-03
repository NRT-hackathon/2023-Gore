#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:08:18 2023



This script chooses 2 random files from the GAN training, and validation sets and finds the closest match using different metrics to the generated structures from the GAN 

"""

from hdf5storage import h5py 
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pathlib
import scipy.io as sio
import GooseEYE
from hdf5storage import loadmat
import pathlib
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error


#Load the training subvolumes obtained from the previous code
pathway = os.chdir('')

#specify which files you want to loop over Validation/Training Set
#pathway = os.chdir('/Users/matt/Desktop/NRT Hackathon/gore2022_train/Subvolumes')

#file_pathway = '/Users/matt/Desktop/NRT Hackathon/gore2022_train/Subvolumes'
file_pathway = '/Users/matt/Desktop/NRT Hackathon/gore2022_val/Subvolumes'

#Get the file names of all the rocks used in the data set
file_pathway_real_rock = '/Users/matt/Desktop/NRT Hackathon/gore2022_val'

filelist = os.listdir(pathway)

filelist_real_rock = os.listdir(file_pathway_real_rock)

# Removing any unneeded files
if 'Processed files' in filelist:
    filelist.remove('Processed files')
    
    
if '.DS_Store' in filelist_real_rock:
    filelist_real_rock.remove('.DS_Store') 

if 'Subvolumes' in filelist_real_rock:
    filelist_real_rock.remove('Subvolumes') 
    
random_files = []


# Generate randomfiles to choose from


sorted_names =[]
pick_temp_store=[]

random_select = 2


# Load 2 random subvolumes from each of the rocks used in the training set.
for j in range(len(filelist_real_rock)):
    pick= (sorted(pathlib.Path('/Users/matt/Desktop/NRT Hackathon/gore2022_val/Subvolumes').glob(filelist_real_rock[j]+'_subvolume'+'*.npy')))
    #pick= (sorted(pathlib.Path('/Users/matt/Desktop/NRT Hackathon/gore2022_val/Subvolumes').glob(filelist_real_rock[j]+'_subvolume'+'*.npy')))
    for k in range(len(pick)):
        pick_temp = str(pick[k])
        #pick_temp_store.append(pick_temp[60:])
        pick_temp_store.append(pick_temp[58:])
        sorted_names.append(pick_temp[58:])
        #pick_temp_store.append(pick_temp[58:])
    for l in range(0, random_select):
       random_file = np.random.choice(pick_temp_store)
       #print(random_file)
       random_files.append(random_file)
    pick_temp_store =[] 
    
    
    
############### Load all of the files associated with the training and validation sets##########

# load everything with respect to the GAN training set   
#os.chdir('/Users/matt/Desktop/NRT Hackathon/Processed files/Sets/Training Set/gore2022_train/Subvolumes/Processed files')
#training_filelist = np.load('Filelist_GAN_train.npy') 
#training_two_point_correlation = np.load('Two_Point_Correlation_GAN_train.npy')
#training_radial_average = np.load('Radial_Average_Profile_GAN_train.npy')


os.chdir('/Users/matt/Desktop/NRT Hackathon/Processed files/Sets/Val Set/gore2022_val/Subvolumes/Processed files')
training_filelist = np.load('Filelist_GAN_val.npy') 
training_two_point_correlation = np.load('Two_Point_Correlation_GAN_val.npy')
training_radial_average = np.load('Radial_Average_Profile_GAN_val.npy')




#Load everything with respect to the GAN generated structures
os.chdir('/Users/matt/Desktop/NRT Hackathon/Processed files/Sets/Archive/Processed files')
GAN_filelist = np.load('Filelist_GAN_Generated.npy')
GAN_two_point_correlation = np.load('Two_Point_Correlation_GAN_Generated.npy')
GAN_radial_average = np.load('Radial_Average_Profile_GAN_Generated.npy')



string_index = []
chunksize =64

#Note: we define x as the true sample input aka the rock and y as the predicted from the GAN
def curve_statistics(x, y):
    if x.shape[0] and y.shape[0] == 1:
        RMSE = mean_squared_error(x,y)
        #euc_distance = np.linalg.norm(x-y)
        return RMSE
    if x.shape[0] and y.shape[0] > 1:
        frobenius_norm = np.linalg.norm((x-y), ord ='fro')
        return frobenius_norm
    


radialavg_comparison = []
radialavg_comparison_calc = np.ones([len(GAN_filelist), len(random_files)])
min_rsme = []
#two_point_comparison = []



#Find the indices of each of the selected random files from the filelist to pull the corresponding metric
for j in range(len(random_files)):
    for k in range(len(training_filelist)):
        if random_files[j] == training_filelist[k]:
            print(k)
            string_index.append(k)
            
for z in range(len(string_index)): 
    print(training_filelist[string_index[z]])
    for h in range(len(GAN_filelist)):
        val = curve_statistics(training_radial_average[0,:,string_index[z]].reshape(1,46), GAN_radial_average[0,:,h].reshape(1,46))
        radialavg_comparison_calc[h,z] = val 
print(radialavg_comparison_calc)

min_index = np.argmin(radialavg_comparison_calc, axis = 0)

 
#Find the minimum RMSE
for i, ind in enumerate(min_index):
    min_rsme.append(radialavg_comparison_calc[ind,i])
min_rsme = np.array(min_rsme)
print(min_index, min_rsme)


min_rmse_table = np.reshape(min_rsme, (2,9))
print(min_rmse_table)

    

  

        
plot1 = training_radial_average[:,:,string_index[z]]
plot2 = GAN_radial_average[:,:, min_index[z]]



g = curve_statistics(plot1,plot2)

fig, ax1 = plt.subplots()
#ax1.plot(plot1[0], label = 'Real Rock:'+ training_filelist[string_index[z]][0:14] + ' Subvolume:' + training_filelist[string_index[z]][-8:-4] , color ='blue',linewidth = 2)
ax1.plot(plot1[0], label = 'Real Rock:'+ training_filelist[string_index[z]][0:14] + ' Subvolume:' + training_filelist[string_index[z]][-7:-4] , color ='blue',linewidth = 2)
ax1.plot(plot2[0], label ='GAN Structure #:'+ GAN_filelist[min_index[z]][4:len(GAN_filelist[min_index[z]])-5], color = 'black', linestyle = 'dotted', linewidth = 2)
ax1.set_ylabel('Radial Averaged 2-Pt Corr', fontsize=14)
ax1.set_xlabel('Radial Distance [Pixels]', fontsize=14)
ax1.text(0.25, 0.5, 'RMSE: {:.2E}'.format(g),  transform=ax1.transAxes,fontsize = 14)
ax1.tick_params(axis='x', labelsize=14) 
ax1.tick_params(axis='y', labelsize=14) 
plt.legend(loc=(0.25, 0.85))
plt.tight_layout()
#plt.savefig('/Users/matt/Desktop/NRT Hackathon/Final Presentation Figures/Closest_match_val.png',dpi=300)




#################### Plotting the structure comparisons ##################### 


#Switch directories to find the subvolumes in the GAN generated structures and the subvolumes
pathway = os.chdir('')
#training_rock_plot = np.load(filelist[string_index[z]])
training_rock_plot = np.load(training_filelist[string_index[z]])


#Gan structure pathway
pathway = os.chdir('')

f= h5py.File(GAN_filelist[min_index[z]], 'r')
GAN_Structure_load = f.get('data')
GAN_structure_plot = np.array(GAN_Structure_load)



'''
fig = plt.figure()
ax=fig.add_subplot(1, 2, 1,projection='3d')
ax.voxels(GAN_structure_plot, facecolors='oldlace', edgecolor='k', linewidth=0.2)
ax.title.set_text('GAN Structure #:' + GAN_filelist[min_index[z]][4:len(GAN_filelist[min_index[z]])-5])

ax=fig.add_subplot(1,2, 2,projection='3d')
ax.voxels(training_rock_plot, facecolors='oldlace', edgecolor='k', linewidth=0.2)
ax.title.set_text('Real Rock:'+ training_filelist[string_index[z]][0:14] + ' Subvolume:' + training_filelist[string_index[z]][-8:-4])
fig.tight_layout()
fig.set_size_inches(10.5, 10.5)
plt.savefig('/Users/matt/Desktop/NRT Hackathon/Final Presentation Figures/Matching_Rocks_val.png',dpi=300)
'''




#plot all the RMSE's for the rock types






fig, ax1 = plt.subplots()
for z in np.arange(0,18,2):
    plot1 = training_radial_average[:,:,string_index[z]]
    plot2 = GAN_radial_average[:,:, min_index[z]]
    g = curve_statistics(plot1,plot2)
    
    h=ax1.plot(plot1[0], label = 'Real Rock:' + training_filelist[string_index[z]][0:14] + ' Subvolume:'   + training_filelist[string_index[z]][-7:-4] ,  linewidth = 2)
    #h=ax1.plot(plot1[0], label ='Real Rock:'
           #   + training_filelist[string_index[z]][0:14] 
            #   + ' Subvolume:' 
             #  + training_filelist[string_index[z]][-8:-4] ,linewidth = 2)
    ax1.plot(plot2[0], label ='GAN Structure #:'
             + GAN_filelist[min_index[z]][4:len(GAN_filelist[min_index[z]])-5]
             + ' RMSE: {:.2E}'.format(g), 
            color = h[0].get_color(), linestyle = 'dotted', linewidth = 2)
    
    ax1.set_ylabel('Radial Averaged 2-Pt Corr', fontsize=14)
    ax1.set_xlabel('Radial Distance [Pixels]', fontsize=14)
    #ax1.text(0.25, 0.5, 'RMSE: {:.2E}'.format(g),  transform=ax1.transAxes,fontsize = 14)
    ax1.tick_params(axis='x', labelsize=14) 
    ax1.tick_params(axis='y', labelsize=14) 
    
ax1.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fontsize =8)
plt.tight_layout()
fig.set_size_inches(15, 8)
fig.savefig('',dpi=300)





        

        
        


    
    
    

    