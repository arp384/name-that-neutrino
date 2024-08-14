# NTN Phase 2 Processing
## Last Updated 8/5/24
## Author: Andrew Phillips (arp384@drexel.edu)

### Description
Directory for performing all preliminary i3 processing for NTN phase2.

'nue_files.csv' and 'numu_files.csv' contain lists of the desired subruns (individual i3 files) to process. I picked these arbitrarily, other than the fact that I made sure to exclude phase1 subruns.

'phase2_filters.py' contains all of the filters for performing monte-carlo truth labeling, event selection, and DAQ frame selection from files. 


### How to use
The main notebook is hase2_processing.ipynb. Running this notebook will go incrementally through all the files specified by the subruns in nue/numu_files.csv and perform processing. For now I hardcoded in the i3 directories, as well as the desired output directories for all the resultant i3s. I'll go into some more detail on this below.

Every "step" in the i3 filtering results in one or more i3 files with added modules. The first one is the mc labeler, which is contained in the file 'APMCLabeler.py' in this directory. This module applies the new MC labeler to all frames, which is very inefficient, unfortuntately. This step takes the longest amount of time. 

The next filter is 'do_cuts', contained in 'phase2_filters.py'. This filter does a whole bunch of different cuts on the i3 frames, cutting on things like event type, charge ratio, and charge magnitude. This cuts down the sizes of the files significantly.

'extract_daq' does just what it sounds like - it cuts all the daq frames in the files. In addition, it splits the very large i3s into 2 MB sizes, and stores them in subdirectories within OUTDIR, named by the subrun. 

The result of running process_files.py will be many subdirectories containing small i3 files which can then be fed into steamshovel, located for now at /scratch/aphillips/phase2_data.

The log file for the most current run is phase2_log.txt. This holds all the output for the notebook. You can see here the filenames of the original i3s used for the processing, as well as the event count history.
