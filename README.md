# name-that-neutrino repository
## Author: Andrew Phillips
Contained here is a collection of python scripts used my name that neutrino analysis for my senior research project at Drexel University. The folder ```i3_processing``` is where it all begins. This folder is used for processing the raw i3 files from the IceCube cobalt server and extracting the Name that Neutrino events. The reason I've created this is so that the original events can be extracted and have new processing applied to them. The ```phase1_analysis``` folder is used for recreating the original analysis done by Elizabeth Warrick during her MS thesis, and also for exploring different retirement limits in the data. Here you will find scripts for extracting different limits, reducing the data, and for automating the production of figures. ```data_processing``` contains scritps for consolidating zooniverse data files and data files from cobalt. 

A lot of this code can be accredited to the work of Elizabeth Warrick and her original Name that Neutrino analysis. 

Note: Beware of hardcoded paths in some of these scripts - I did my best to get rid of those but it's possible I didn't iron all of them out. 

The main steps to these analyses are as follows.

## Event Selection / Feature Extraction on Cobalt
I did this part of the analysis on the IceCube server, cobalt, but in theory it could be performed on a local machine if you have the NTN phase 1 data files at hand, and the icecube icetray software running on your machine. All of the relevant files for the cobalt server side of things are located in the ```i3_processing``` directory. I'll provide a general overview of all the files located here:

#### ap_modules.py
This is a collection of i3 modules that I used for custom data processing. You'll see in here classes such as ```CorsikaLabeler```, the module I use for applying Corsika (i.e., background) labels to events, and ```Qtot``` which I use for summing all the pulse charge in events. 
#### enums.py
This file contains a bunch of mappings, the details of which aren't super important. E.g., the ```classifications``` map is used to map event types to numbers which are easier to handle with i3 processing. For instance, it maps "throughgoing track" to the number 1. This is semi-important when we are converting back from numbers to event types on the local side.
#### filters.py
These are the filters used in the original generation of the NTN dataset. The file contains a bunch of functions used for event selection. I.e., it contains the functions for generating a uniform energy distribution for events and for isolating DAQ frames. I don't use this file for my current analysis, but it's good to know how the events were chosen initially. 
#### i3_tools.py
As implied by its name, this file contains tools for processing i3 files. Namely, it has functions for applying the modules defined in ```ap_modules.py``` and then for extracting data from generated hd5 files and turning it into csv form. 
#### mc_labeler.py 
This file was inherited from Elizabeth Warrick, and it's used for applying plain MC truth labels (in my words, signal labels) to events. I've modified this file slightly to also compute the signal charge contribution and add it to event frames.
#### phase1_files.csv
This is a file I created that stores all the ntn phase1 data file paths, along with their relevant subject set ids. 
#### do_analysis.ipynb
This notebook is essentially a wrapper for all of these files. It takes in all the ntn file paths, extracts only the event ids we want, does custom processing, and creates a csv containing all the information we want. 

## Reducing NTN data exports

This stuff can be done of the local side, provided you have access to the name that neutrino data exports. The relevant files are located in the ```phase1_analysis``` directory. I'll again walk through each file:


#### get_retired.py
This is a module I wrote for extracting data at a particular retirement limit. 

#### reducer.py
This a data reducer, e.g., it simplifies the massive ntn classification dataset into a simpler form, tabulating subject set ids, event_ids, the winning classification along with its vote tally, and the user agreement fraction. I wrote this because the ```panoptes_aggregation``` library that we were using initially was being annoying on my system so its sort of a workaround. 

#### phase1_data_analysis.py
This is a module used for taking in the previously created reduced data and combining with other subject metadata to tabulate events, user classifications, dnn classifications, along with event energies, zenith, etc. 

#### do_analysis.py
This is a script that wraps everything together. You can use it to run phase1_data_analysis.py on a series of different retirement limits and auto-generate plots. 

## Putting it all together
Once you've done both the local and cobalt side of the data extraction, the next step is to put everything together. This is done with the ```aggregate.py``` script. This takes in the consensus data csv from the local analysis, as well as the mc information from cobalt side, and combines it all into a master csv. The file ```make_plots``` is then used to easy display the results. 


## Phase 2 Processing
TBD

