# NTN Phase 1 Analysis
## Last Updated 4/22/24
## Author: Andrew Phillips (arp384@drexel.edu)

### Description
The purpose of these scripts is to process raw data exports from the Name that
Neutrino website and apply filters for custom retirement limits.
Given the raw classification and subject data, these programs will reconstruct what
the dataset would have looked like after being retired at some previous retirement
limit. For example, if the zooniverse retirement limit is 40 but you wish to make
plots for a retirement limit of 15, these programs will cut the dataset at a past
date where every subject was classified exactly 15 times. 

### How to use
The first thing you need to do is download the raw subject, classification, and workflow datasets from the name that neutrino page. You can find those at this link: https://www.zooniverse.org/projects/icecubeobservatory/name-that-neutrino under the 'lab' tab. Note that you'll need administrative access to see that tab, so a good first step would be to set up an account here. 

There's a few different python files in this directory, do_analysis.py, get_retired.py, reducer.py, and phase1_analysis.py. get_retired.py is used to extract the classification data at the time at which a particular retirement was achieved. For example, if you want to see what the data looked like at the time that every subject was retired 15 times. Basically, this script reads through all the classifications and keeps a hash map of all the subjects. Once it counts 15 classifications for all 4272 subjects, it stops and saves off the result as a new classification set. The file reducer.py contains the Reducer class: a tool to extract only the relevant information from our ntn data exports. This will give a new csv tabulating the subject id of every subject along with the user consensus choice, and the fraction agreement. phase1_data_analysis.py contains a few tools for performing analysis of the resultant dataset, including functions to consolidate user, dnn, and mc data, and for making plots. 

do_analysis.py is the main script here. It'll do everything in one shot for you, compiling the retired data, reducing it, and saving figures for you. To run it, just execute the following on the command line:

'python do_analysis.py <retirement_limit> <exports_directory>'

'retirement_limit' is the retirement limit you want. This can be just one number, or it also recognizes a list of limits and will iterate over all of them for you. 'exports_directory' is the directory containing all of the raw data exports which you should have already downloaded from zooniverse (Those should look something like name-that-neutrino-classifications.csv, name-that-neutrino-subjects.csv, name-that-neutrino-workflows.csv). The script will create a new directory called data_{retirement_lim}lim where it'll put all the output. Plots will be saved inside that directory within another subdirectory called 'plots'. 



