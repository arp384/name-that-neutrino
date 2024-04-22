Last Updated 4/3/24
Author: Andrew Phillips (arp384@drexel.edu)

The purpose of these scripts is to process raw data exports from the Name that
Neutrino website and apply filters for custom retirement limits.
Given the raw classification and subject data, these programs will reconstruct what
the dataset would have looked like after being retired at some previous retirement
limit. For example, if the zooniverse retirement limit is 40 but you wish to make
plots for a retirement limit of 15, these programs will cut the dataset at a past
date where every subject was classified exactly 15 times. 

A python version of at least 3.0 and a few packages are required for this code 
to work. You will have to install numpy, matplotlib, pandas, seaborn, os, sys, json,
and argparse. 

If you just want to skip to making plots and don't feel like going through all the steps 
there's a handy dandy windows .bat file that will do everything for you. You just have to execute:
	makeplots.bat <retirement limit> <location of raw ntn data exports>
Where the two arguments are the retirement limit you want and the location of your data exports directly from zooniverse.
This will make a bunch of new files, but most importantly will populate a new directory data_15lim/plots with your new figures.

Otherwise here's the following are all steps required to run this project. (Note: this was written on windows,
but in principle should work on linux). **All paths must be expressed relative to CWD!! (e.g., "data_15lim" as opposed to "home\...\...\data_15lim")
If you try absolute paths it WILL NOT WORK!!!!

1. Download name that neutrino classification, subject, and workflow datasets from
https://www.zooniverse.org/lab/19023/data-exports. Note that you'll need administrative
access to the project in order for this page to appear. Place these files within
a subdirectory in this folder.

2. Make sure you are in the root phase1_data_analysis directory.

3. In the command line, type the following command:
	python get_retired.py <limit> <input_dir> <output_dir>
Where <limit> denotes the desired retirement limit, <input_dir> is the directory
containing the name that neutrino exports you just downloaded, and <output_dir>
is the directory where you want the resulting files to be stored. 

4. Next we want to reduce the user data. To do so run the command 
	python reducer.py <lim> <input_dir> <output_dir>
Where <input_dir> is the directory you createdin step 3, and <output_dir> is 
where you want the result to go. This will put a new file consensus_reduced.csv in
the output directory.

6. To make plots, run the following command:
	python phase1_data_analysis.py <lim> <input_dir> <output_dir>
Where <lim> is again your desired retirement limit, <input_dir> is the 
directory you made in step 3, and <output_dir> is the directory where you want
the resultant files to go. 

7. <output_dir> should now contain a new directory called "plots" with the resulting 
plots, as well as a new file ntn_result_consensus.csv containing the aggregated
data. 




