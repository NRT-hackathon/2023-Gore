# 2023-Gore

To run the code in the repository the following Python packages are required:

1) Numpy
2) PoreSpy
3) GooseEye
4) Scikit learn
5) Os
6) Matplotlib
7) hdf5storage
8) pathlib

The script files accomplish the follow tasks and are to be run in the following order. 

1) Generate_Subvolumes_upload.py: Given the raw input files from the digital rocks portal (Project 374 in this case) this code takes each of the downloaded files
and divides the cube into subvolumes that are 64 cubic voxels in size with at least two pore values in each of the structures. If this requirement is achieved,
the subvolume is saved with the corresponding filename and subvolume number (global).

2) Gan_Structure_Metrics_Upload.py: Takes a filelist of inputs for the structures generated in the previous code. Specifically, the code calculates for each file 
the porosity, porosity profile, (across each of the dimensions) as well as the two point correlation function (Note: GooseEYE does not normalize properly, measurements above ~32 pixels may be biased). Each of these metrics is stored as a numpy array and
saved with the appropriate filename. 

3) Calculate_Statistics_Upload.py: Uses the array of values saved from the previous code and calculates the average, standard deviation, and variance for each of the 
metrics. Using these values, the rest of the code plots histograms and various curves with error bars for each of the metrics. 


4) Local_Structure_Comparison_Upload.py: Used to find the closest matching GAN_structure from the fileset list. This code takes in all the metrics from (2) and randomly
chooses two subvolumes from the rock types loaded in the training and validation data set. Looping over for each of the random files all of the GAN structures, 
the root mean square error for the radially averaged two-point correlation function is calculated between each file. The closest match is found by taking the minimum root 
mean square error across all the GAN structures per each file in the data set. The code then finds the particular rock subvolumes for the closest match and plots the voxel and 
two point correlation representation. 
