#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:16:17 2023

"""

from hdf5storage import h5py 
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from hdf5storage import loadmat
import scipy.io as sio


#Specify the directory where the files are saved, the pathway of where to save the files and their respective pathway
pathway = os.chdir('')
file_pathway = ''
save_pathway = ''

filelist = os.listdir(pathway)

#Remove any files not to be used in the analysis
if '.DS_Store' in filelist:
    filelist.remove('.DS_Store')

if 'Subvolumes' in filelist:
    filelist.remove('Subvolumes')


print(filelist)

poresrequired = 2
chunksize = 64 
count = 0


for index in range(len(filelist)):
    itm =  loadmat(os.path.join(file_pathway, filelist[index]))['bin']
    file = filelist[index]
    print(file)
    print(index)

    for newx in range(itm.shape[0]//chunksize):
        for newy in range(itm.shape[1]//chunksize):
            for newz in range(itm.shape[2]//chunksize):
                subvolumes = itm[chunksize*newx:chunksize*(newx+1),chunksize*newy:chunksize*(newy+1),chunksize*newz:chunksize*(newz+1)]
                num_zeros = np.sum(subvolumes==0)
                if num_zeros > poresrequired:
                    count  +=1
                    np.save(f'{save_pathway}/{os.path.basename(filelist[index]).split()[0]}_subvolume{count:02}', subvolumes)

 

#Plot one of the cube subvolumes
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(subvolumes, facecolors='oldlace', edgecolor='k', linewidth=0.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

           