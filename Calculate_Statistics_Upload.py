#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:12:26 2023




"""


import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import seaborn as sns
import scipy.stats




#Load the file pathway where everything is saved 
directory_pathway = os.chdir('')
filelist = os.listdir(directory_pathway)

# Remove any hidden files or folders that are not necessary
if 'Sets' or '.DS_Store' in filelist:
    filelist.remove('Sets')
    filelist.remove('.DS_Store')
   




chunksize = 64

center = chunksize/2 * np.ones((1,2)) #This has to be of shape (2,1) otherwise the matrix dimensions do not match up 
dmax =np.sqrt((0-center[0,0])**2 + (0-center[0,1])**2)
dmax = dmax.astype(np.int)

# Returns a tuple of summary statistics for each of the generated structures
def get_stats(x):
    if x.ndim == 3:
        variance = np.var(x, axis =2)
        standard_deviation = np.std(x, axis =2)
        mean = np.mean(x, axis = 2)
        ub = mean + standard_deviation
        lb = mean - standard_deviation
    else:  
        variance = np.var(x)
        standard_deviation = np.std(x)
        mean = np.mean(x)
        ub = mean + standard_deviation
        lb = mean - standard_deviation
    return mean, lb, ub, standard_deviation, variance


# Load all the files as a variable 
for i in range(len(filelist)):
    vars()[filelist[i][0:-4]] = np.load(filelist[i])
    vars()[filelist[i][0:-4]+'_stats'] = get_stats(vars()[filelist[i][0:-4]])
    
    
xplot = np.linspace(0, chunksize,chunksize)
radius_plot = np.linspace(0, dmax+1, dmax+1)


######################### Generating Figures##############################
'''
plt.figure()
plt.plot(xplot, porosity_x_profile_GAN_stats[0][0,:])
plt.fill_between(xplot,porosity_x_profile_GAN_stats[1][0,:] ,porosity_x_profile_GAN_stats[2][0,:], alpha=0.2, label='error band')
plt.plot(xplot, porosity_y_profile_GAN_stats[0][0,:])
plt.fill_between(xplot,porosity_y_profile_GAN_stats[1][0,:] ,porosity_y_profile_GAN_stats[2][0,:], alpha=0.2, label='error band')
plt.plot(xplot, porosity_z_profile_GAN_stats[0][0,:])
plt.fill_between(xplot,porosity_z_profile_GAN_stats[1][0,:] ,porosity_z_profile_GAN_stats[2][0,:], alpha=0.2, label='error band')
plt.xlabel('Distance [Pixels]')
plt.ylabel('Average Porosity')
plt.show()
'''


fig, axs = plt.subplots(3, sharex=True, sharey=True)
axs[0].plot(xplot, porosity_x_profile_GAN_stats[0][0,:], label = 'Mean')
axs[0].fill_between(xplot,porosity_x_profile_GAN_stats[1][0,:] ,porosity_x_profile_GAN_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[0].plot(xplot, porosity_x_profile_GAN_test_stats[0][0,:], label = 'Mean')
axs[0].fill_between(xplot,porosity_x_profile_GAN_test_stats[1][0,:] ,porosity_x_profile_GAN_test_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[0].plot(xplot, porosity_x_profile_GAN_val_stats[0][0,:], label = 'Mean')
axs[0].fill_between(xplot,porosity_x_profile_GAN_val_stats[1][0,:] ,porosity_x_profile_GAN_val_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[0].plot(xplot, porosity_x_profile_GAN_train_stats[0][0,:], label = 'Mean')
axs[0].fill_between(xplot,porosity_x_profile_GAN_train_stats[1][0,:] ,porosity_x_profile_GAN_train_stats[2][0,:], alpha=0.2, label='Standard Deviation')
axs[0].set_title('Porosity Profile yz-plane')


axs[1].plot(xplot, porosity_y_profile_GAN_stats[0][0,:], label = 'Mean')
axs[1].fill_between(xplot,porosity_y_profile_GAN_stats[1][0,:] ,porosity_y_profile_GAN_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[1].plot(xplot, porosity_y_profile_GAN_test_stats[0][0,:], label = 'Mean')
axs[1].fill_between(xplot,porosity_y_profile_GAN_test_stats[1][0,:] ,porosity_y_profile_GAN_test_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[1].plot(xplot, porosity_y_profile_GAN_val_stats[0][0,:], label = 'Mean')
axs[1].fill_between(xplot,porosity_y_profile_GAN_val_stats[1][0,:] ,porosity_y_profile_GAN_val_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[1].plot(xplot, porosity_y_profile_GAN_train_stats[0][0,:], label = 'Mean')
axs[1].fill_between(xplot,porosity_y_profile_GAN_train_stats[1][0,:] ,porosity_y_profile_GAN_train_stats[2][0,:], alpha=0.2, label='Standard Deviation')
axs[1].set_title('Porosity Profile xz-plane')



axs[2].plot(xplot, porosity_z_profile_GAN_stats[0][0,:], label = 'Mean')
axs[2].fill_between(xplot,porosity_z_profile_GAN_stats[1][0,:] ,porosity_z_profile_GAN_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[2].plot(xplot, porosity_z_profile_GAN_test_stats[0][0,:], label = 'Mean')
axs[2].fill_between(xplot,porosity_z_profile_GAN_test_stats[1][0,:] ,porosity_z_profile_GAN_test_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[2].plot(xplot, porosity_z_profile_GAN_val_stats[0][0,:], label = 'Mean')
axs[2].fill_between(xplot,porosity_z_profile_GAN_val_stats[1][0,:] ,porosity_z_profile_GAN_val_stats[2][0,:], alpha=0.2, label='Standard Deviation')

axs[2].plot(xplot, porosity_z_profile_GAN_train_stats[0][0,:], label = 'Mean')
axs[2].fill_between(xplot,porosity_z_profile_GAN_train_stats[1][0,:] ,porosity_z_profile_GAN_train_stats[2][0,:], alpha=0.2, label='Standard Deviation')
axs[2].set_title('Porosity Profile xy-plane')

fig.supxlabel('Distance [Pixels]')
fig.supylabel('Porosity')

plt.tight_layout()
plt.show()




fig, axs = plt.subplots(3, sharex=True, sharey=True)
axs[0].plot(xplot, porosity_x_profile_GAN_stats[0][0,:], label = 'Mean Porosity Profile GAN Generated')
axs[0].fill_between(xplot,porosity_x_profile_GAN_stats[1][0,:] ,porosity_x_profile_GAN_stats[2][0,:], alpha=0.2, label='Standard Deviation GAN Generated')


axs[0].plot(xplot, porosity_x_profile_GAN_train_stats[0][0,:], label = 'Mean Porosity Profile Training Set')
axs[0].fill_between(xplot,porosity_x_profile_GAN_train_stats[1][0,:] ,porosity_x_profile_GAN_train_stats[2][0,:], alpha=0.2, label='Standard Deviation Training Set')
axs[0].set_title('Porosity Profile yz-plane')

axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

axs[1].plot(xplot, porosity_y_profile_GAN_stats[0][0,:])
axs[1].fill_between(xplot,porosity_y_profile_GAN_stats[1][0,:] ,porosity_y_profile_GAN_stats[2][0,:], alpha=0.2)


axs[1].plot(xplot, porosity_y_profile_GAN_train_stats[0][0,:])
axs[1].fill_between(xplot,porosity_y_profile_GAN_train_stats[1][0,:] ,porosity_y_profile_GAN_train_stats[2][0,:], alpha=0.2)
axs[1].set_title('Porosity Profile xz-plane')



axs[2].plot(xplot, porosity_z_profile_GAN_stats[0][0,:])
axs[2].fill_between(xplot,porosity_z_profile_GAN_stats[1][0,:] ,porosity_z_profile_GAN_stats[2][0,:], alpha=0.2)


axs[2].plot(xplot, porosity_z_profile_GAN_train_stats[0][0,:])
axs[2].fill_between(xplot,porosity_z_profile_GAN_train_stats[1][0,:] ,porosity_z_profile_GAN_train_stats[2][0,:], alpha=0.2)
axs[2].set_title('Porosity Profile xy-plane')

fig.supxlabel('Distance [Pixels]')
fig.supylabel('Porosity')

plt.tight_layout()
plt.show()




##### Plotting the Porosity ##############
xspace = np.arange(0, 1, 0.2)
yspace = np.arange(0,4, 1)
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(porosity_GAN,density = True)
axs[0, 0].set_title('GAN Generated Structures')
axs[0,0].set_xlabel('Porosity', fontsize = 14)
axs[0,0].set_ylabel('Frequency', fontsize = 14)
axs[0,0].tick_params(axis='x', labelsize=14) 
axs[0,0].tick_params(axis='y', labelsize=14) 
axs[0,0].set_xticks(xspace)
axs[0,0].set_yticks(yspace)

axs[0, 1].hist(porosity_GAN_test, density = True)
axs[0, 1].set_title('GAN Test Set')
axs[0,1].set_xlabel('Porosity', fontsize = 14)
axs[0,1].set_ylabel('Frequency', fontsize = 14)
axs[0,1].tick_params(axis='x', labelsize=14) 
axs[0,1].tick_params(axis='y', labelsize=14) 
axs[0,1].set_xticks(xspace)
axs[0,1].set_yticks(yspace)


axs[1, 0].hist(porosity_GAN_train, density = True)
axs[1, 0].set_title('GAN Training Set')
axs[1,0].set_xlabel('Porosity', fontsize = 14)
axs[1,0].set_ylabel('Frequency', fontsize = 14)
axs[1,0].tick_params(axis='x', labelsize=14) 
axs[1,0].tick_params(axis='y', labelsize=14) 
axs[1,0].set_xticks(xspace)
axs[1,0].set_yticks(yspace)

axs[1, 1].hist(porosity_GAN_val,density = True)
axs[1, 1].set_title('GAN Validation Set')
axs[1,1].set_xlabel('Porosity', fontsize = 14)
axs[1,1].set_ylabel('Frequency', fontsize = 14)
axs[1,1].tick_params(axis='x', labelsize=14) 
axs[1,1].tick_params(axis='y', labelsize=14) 
axs[1,1].set_xticks(xspace)
axs[1,1].set_yticks(yspace)
plt.tight_layout()
 
plt.show()
fig.set_size_inches(10.5, 10.5)
#plt.savefig('Porosity_Distribution_Subplot',dpi=300)

################# Plotting the Solid Phase Distribution###########
xspace = np.arange(0, 1, 0.2)
yspace = np.arange(0,4, 1)
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist((1-porosity_GAN),density = True)
axs[0,0].axvline((1-porosity_GAN).mean(), color ='k', linestyle='dashed', linewidth=1)
axs[0,0].text((1-porosity_GAN).mean()*0.05,0.9, 'Mean: {:.2f}'.format((1-porosity_GAN).mean()))
axs[0, 0].set_title('GAN Generated Structures')
axs[0,0].set_xlabel('Solid Phase Fraction', fontsize = 14)
axs[0,0].set_ylabel('Density', fontsize = 14)
axs[0,0].tick_params(axis='x', labelsize=14) 
axs[0,0].tick_params(axis='y', labelsize=14) 
axs[0,0].set_xticks(xspace)
axs[0,0].set_yticks(yspace)

axs[0, 1].hist((1-porosity_GAN_test), density = True)
axs[0,1].axvline((1-porosity_GAN_test).mean(), color ='k', linestyle='dashed', linewidth=1)
axs[0,1].text((1-porosity_GAN_test).mean()*0.05,0.9, 'Mean: {:.2f}'.format((1-porosity_GAN_test).mean()))
axs[0, 1].set_title('GAN Test Set')
axs[0,1].set_xlabel('Solid Phase Fraction', fontsize = 14)
axs[0,1].set_ylabel('Density', fontsize = 14)
axs[0,1].tick_params(axis='x', labelsize=14) 
axs[0,1].tick_params(axis='y', labelsize=14) 
axs[0,1].set_xticks(xspace)
axs[0,1].set_yticks(yspace)


axs[1, 0].hist((1-porosity_GAN_train), density = True)
axs[1,0].axvline((1-porosity_GAN_train).mean(), color ='k', linestyle='dashed', linewidth=1)
axs[1,0].text((1-porosity_GAN_train).mean()*0.05,0.9, 'Mean: {:.2f}'.format((1-porosity_GAN_train).mean()))
axs[1, 0].set_title('GAN Training Set')
axs[1,0].set_xlabel('Solid Phase Fraction', fontsize = 14)
axs[1,0].set_ylabel('Density', fontsize = 14)
axs[1,0].tick_params(axis='x', labelsize=14) 
axs[1,0].tick_params(axis='y', labelsize=14) 
axs[1,0].set_xticks(xspace)
axs[1,0].set_yticks(yspace)

axs[1, 1].hist((1-porosity_GAN_val),density = True)
axs[1,1].axvline((1-porosity_GAN_val).mean(), color ='k', linestyle='dashed', linewidth=1)
axs[1,1].text((1-porosity_GAN_val).mean()*0.05,0.9, 'Mean: {:.2f}'.format((1-porosity_GAN_val).mean()))
axs[1, 1].set_title('GAN Validation Set')
axs[1,1].set_xlabel('Solid Phase Fraction', fontsize = 14)
axs[1,1].set_ylabel('Density', fontsize = 14)
axs[1,1].tick_params(axis='x', labelsize=14) 
axs[1,1].tick_params(axis='y', labelsize=14) 
axs[1,1].set_xticks(xspace)
axs[1,1].set_yticks(yspace)
plt.tight_layout()
#plt.savefig('/Users/matt/Desktop/NRT Hackathon/Final Presentation Figures/Solid_Phase_Histogram.png',dpi=300)




############Overlapping Histograms####################
plt.figure()
plt.hist(porosity_GAN, alpha = 0.2, label = 'GAN Generated')
plt.hist(porosity_GAN_test, alpha = 0.2, label = 'GAN Test Set')
plt.hist(porosity_GAN_train, alpha =0.2, label = 'GAN Training Set')
plt.hist(porosity_GAN_val, alpha=0.2, label = 'GAN Validation Set')
plt.xlabel('Porosity', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.tight_layout()
#plt.savefig('Porosity_Distribution_Transparent histogram',dpi=300)
bbox_inches='tight'
plt.show()


fig, axs = plt.subplots(1, 2)
axs[0].hist(1-porosity_GAN_test, density = True, label = 'GAN Test Set', color ='green')
axs[0].hist(1-porosity_GAN_train,density = True, label = 'GAN Training Set', color = 'blue')
axs[0].hist(1-porosity_GAN_val, density = True, label = 'GAN Validation Set', color = 'orange')
axs[0].hist(1-porosity_GAN, density = True, label = 'GAN Generated', color = 'red')
axs[0].set_xlabel('Solid Phase', fontsize = 14)
axs[0].set_ylabel('Density', fontsize = 14)
axs[0].set_xticks(xspace, fontsize=14)
axs[0].set_yticks(yspace, fontsize=14)
axs[0].legend()

axs[1].hist(1-porosity_GAN,alpha = 0.2,  density = True, label = 'GAN Generated', color = 'red')
axs[1].hist(1-porosity_GAN_test,alpha = 0.2, density = True, label = 'GAN Test Set', color ='green')
axs[1].hist(1-porosity_GAN_train,alpha = 0.2,density = True, label = 'GAN Training Set', color = 'blue')
axs[1].hist(1-porosity_GAN_val,alpha = 0.2, density = True, label = 'GAN Validation Set', color = 'orange')
axs[1].set_xlabel('Solid Phase', fontsize = 14)
axs[1].set_ylabel('Density', fontsize = 14)
axs[1].set_xticks(xspace, fontsize=14)
axs[1].set_yticks(yspace, fontsize=14)
axs[1].legend()

fig.set_size_inches(10.5,8)
fig.savefig('/Users/matt/Desktop/NRT Hackathon/Final Presentation Figures/Solid_Phase_Distribution_Transparent histogram.png',dpi=300)










############################################################
plt.figure()
plt.plot(radius_plot, Radial_Average_Profile_GAN_Generated_stats[0][0,:], label='GAN Generated Structures')
plt.fill_between(radius_plot,Radial_Average_Profile_GAN_Generated_stats[1][0,:], Radial_Average_Profile_GAN_Generated_stats[2][0,:], alpha = 0.2, label ='GAN Generated Stdev')
plt.yticks(np.linspace(0.5, 1, 10))
#plt.plot(radius_plot, Radial_Average_Profile_GAN_test_stats[0][0,:], label ='GAN Test Set')
#plt.fill_between(radius_plot,Radial_Average_Profile_GAN_test_stats[1][0,:], Radial_Average_Profile_GAN_test_stats[2][0,:], alpha = 0.2, label = ' Test Set Stdev')

#plt.plot(radius_plot, Radial_Average_Profile_GAN_val_stats[0][0,:], label ='Validation Set')
#plt.fill_between(radius_plot,Radial_Average_Profile_GAN_val_stats[1][0,:], Radial_Average_Profile_GAN_val_stats[2][0,:], alpha = 0.2, label='Validation Set Stdev')

#plt.plot(radius_plot, Radial_Average_Profile_GAN_train_stats[0][0,:], label = 'GAN Train Set')
#plt.fill_between(radius_plot,Radial_Average_Profile_GAN_train_stats[1][0,:], Radial_Average_Profile_GAN_train_stats[2][0,:], alpha = 0.2, label = 'Train Set Stdev')
plt.ylabel('Radially Averaged Two-Point Correlation', fontsize = 14)
plt.xlabel('Radial Distance [Pixels]', fontsize = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()


'''
plt.figure()
plt.imshow(Two_Point_Correlation_GAN_test[:,:,0])
plt.colorbar()
plt.savefig('2p1',dpi=300)

plt.figure()
plt.imshow(Two_Point_Correlation_GAN_test[:,:,20])
plt.colorbar()
plt.savefig('2p2',dpi=300)
            
plt.figure()
plt.imshow(Two_Point_Correlation_GAN_test[:,:,100])
plt.colorbar()
plt.savefig('2p3',dpi=300)

plt.figure()
plt.plot(radius_plot, Radial_Average_Profile_GAN_Generated_stats[0][0,:])
plt.fill_between(radius_plot, Radial_Average_Profile_GAN_Generated_stats[1][0,:],Radial_Average_Profile_GAN_Generated_stats[2][0,:], alpha = 0.2)
plt.xlabel('Radius [pixels]')
plt.ylabel('Two-Point Correlation')
plt.show()'
'''


    
    
    