#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:03:32 2023


"""


'''
Note: This code takes and analyzes the structures from the GAN outputs in hdf5 format. 
Dataset home: https://www.digitalrocksportal.org/projects/374

Note: Data set considers pore-space with zeros and solid with one's

Porespy considers 1's as void space and zero's as solid phase so make sure to subtract one

GooseEye returns the probability that both points are white (solid in this case). So probabiliy porespace = 1-

'''


from hdf5storage import h5py 
import numpy as np
import matplotlib.pyplot as plt
import os
import porespy as ps
import time
import GooseEYE
from tqdm import tqdm
import inspect
inspect.signature(ps.metrics.two_point_correlation)



def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0,0])**2 + (y - center[0,1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile



directory_pathway = os.chdir('') # Directory where the files are stored
filelist = os.listdir(directory_pathway) # List all the files in the directory pathway, check for hidden files
save_pathway = ''

if '.DS_Store' in filelist or 'Processed files' in filelist: #Remove any hidden files made from processing/importing the data
    filelist.remove('.DS_Store')
    filelist.remove('Processed files')
    
    
  
print(filelist)

chunksize = 64




center = chunksize/2 * np.ones((1,2)) #This has to be of shape (2,1) otherwise the matrix dimensions do not match up 
dmax =np.sqrt((0-center[0,0])**2 + (0-center[0,1])**2)
dmax = dmax.astype(np.int)


#Pre-allocating storage for different variables
porosity = np.zeros(len(a))
porosity_x_profile = np.zeros((1, chunksize, len(a)))
porosity_y_profile = np.zeros((1, chunksize, len(a)))
porosity_z_profile = np.zeros((1, chunksize, len(a)))

two_point_correlation_store =np.zeros((chunksize+1, chunksize+1, chunksize))


two_point_correlation = np.zeros((chunksize+1, chunksize+1, len(a)))
radial_profile_store = np.zeros((1, dmax+1, len(a)))
count = 0


#######################Looping over all the data################################
for i in range(len(a)):
    #file_load= h5py.File(filelist[i], 'r')
    file_load = np.load(filelist[i])
    if i% 50 == 0:
        print(i)
        print(filelist[i])
#Debugging, check the plot to make sure this makes physical sense
    #rock_structure = np.array(file_load.get('data'))
    rock_structure = file_load
    porosity[i] = 1-(ps.metrics.porosity(rock_structure)) #subtract one because porespy considers 1's as void and 0's as solid
    
    #Create temporary variables to pass onto the storage arrays below
    porosity_x_profile_temp = np.reshape((1-ps.metrics.porosity_profile(rock_structure, axis = 0)), (1, chunksize))
    porosity_y_profile_temp = np.reshape((1-ps.metrics.porosity_profile(rock_structure, axis = 1)), (1, chunksize))
    porosity_z_profile_temp = np.reshape((1-ps.metrics.porosity_profile(rock_structure, axis = 2)), (1, chunksize))
    
    #Store the first row, all the columns in the ith 3rd dimension of the respective axis profiles
    porosity_x_profile[0,:,i] = porosity_x_profile_temp
    porosity_y_profile[0,:,i] = porosity_y_profile_temp
    porosity_z_profile[0,:,i] = porosity_z_profile_temp

    for j in range(chunksize):
        #tic = time.time()
        two_point_correlation_store[:,:,j] = GooseEYE.S2((chunksize+1,chunksize+1), rock_structure[:,:,j], rock_structure[:,:,j]) # Probability of a solid 
        #toc = time.time()
        #print(toc-tic)
        #if j%30 == 0:
         #  print(j)
          #  plt.figure()
           # plt.imshow(two_point_correlation_store[:,:,j])
           # plt.show()
          
    average_two_point_correlation = np.mean(two_point_correlation_store, axis = 2)
    two_point_correlation[:,:,i] = average_two_point_correlation
    radial_prof = radial_profile(average_two_point_correlation, center)
    radial_prof = np.reshape(radial_prof, (1, dmax+1))
    radial_profile_store[0,:,i]=radial_prof
   





#Save the corresponding files to be used for later in the analysis
np.save('', two_point_correlation) # Save the file list to this pathway
np.save('', radial_profile_store) # Save the file list to this pathway
np.save('', filelist) # Save the file list to this pathway
np.save('', porosity_x_profile) # Save the file list to this pathway
np.save('', porosity_y_profile) # Save the file list to this pathway
np.save('', porosity_z_profile) # Save the file list to this pathway
np.save('', porosity) # Save the file list to this pathway

################################################################################# 
   

'''
##################### Calculating Statistics of the profiles #######################    
porosity_x_profile_average = np.mean(porosity_x_profile, axis = 2)
porosity_x_profile_std = np.std(porosity_x_profile, axis = 2)
porosity_x_profile_var = np.var(porosity_x_profile, axis = 2)

porosity_y_profile_average = np.mean(porosity_y_profile, axis = 2)
porosity_y_profile_std = np.std(porosity_y_profile, axis = 2)
porosity_y_profile_var = np.var(porosity_y_profile, axis = 2)


porosity_z_profile_average = np.mean(porosity_z_profile, axis = 2)
porosity_z_profile_std = np.std(porosity_z_profile, axis = 2)
porosity_z_profile_var = np.var(porosity_z_profile, axis = 2)



average_porosity = np.mean(porosity)
stdev_porosity = np.std(porosity)
variance_porosity = np.var(porosity)

average_radial_profile = np.mean(radial_profile_store, axis =2)
stdev_radial_profile = np.std(radial_profile_store, axis =2)
var_radial_profile = np.var(radial_profile_store, axis =2)

##################################################################################

# Compute distances to plot over 
x_distance = np.linspace(0, porosity_x_profile_average.shape[1], porosity_x_profile_average.shape[1] )
y_distance = np.linspace(0, porosity_y_profile_average.shape[1], porosity_y_profile_average.shape[1] )
z_distance = np.linspace(0, porosity_z_profile_average.shape[1], porosity_z_profile_average.shape[1] )



#plot the average porosity distribution along each of the principle axis
plt.figure()
plt.plot(x_distance, np.transpose(porosity_x_profile_average.reshape(-1)))
plt.plot(y_distance, np.transpose(porosity_y_profile_average.reshape(-1)))
plt.plot(z_distance, np.transpose(porosity_z_profile_average.reshape(-1)))
plt.show()


# Now include the statistics with the error bars 
plt.figure()
plt.errorbar(x_distance,np.transpose(porosity_x_profile_average.reshape(-1)) , yerr = np.transpose(porosity_x_profile_std.reshape(-1)), fmt ='o' ,label='yz-plane')
plt.errorbar(y_distance,np.transpose(porosity_y_profile_average.reshape(-1)) , yerr = np.transpose(porosity_y_profile_std.reshape(-1)), fmt ='o', label='xz-plane')
plt.errorbar(z_distance,np.transpose(porosity_z_profile_average.reshape(-1)) , yerr = np.transpose(porosity_z_profile_std.reshape(-1)), fmt ='o', label='xy-plane')
plt.xlabel('Distance [pixels]')
plt.ylabel('Average Porosity')
plt.legend()
plt.show()

plt.figure()
plt.hist(porosity)
plt.xlabel('Porosity')
plt.ylabel('Frequency')
'''
