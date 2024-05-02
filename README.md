# name-that-neutrino repository
## Author: Andrew Phillips
Contained here is a collection of python scripts used my name that neutrino analysis for my senior research project at Drexel University. The folder ```i3_processing``` is where it all begins. This folder is used for processing the raw i3 files from the IceCube cobalt server and extracting the Name that Neutrino events. The reason I've created this is so that the original events can be extracted and have new processing applied to them. The ```phase1_analysis``` folder is used for recreating the original analysis done by Elizabeth Warrick during her MS thesis, and also for exploring different retirement limits in the data. Here you will find scripts for extracting different limits, reducing the data, and for automating the production of figures. ```data_processing``` contains scritps for consolidating zooniverse data files and data files from cobalt. 

A lot of this code can be accredited to the work of Elizabeth Warrick and her original Name that Neutrino analysis. 

The main steps to these analyses are as follows.

### Event Selection / Feature Extraction on Cobalt
I did this part of the analysis on the IceCube server, cobalt, but in theory it could be performed on a local machine if you have the NTN phase 1 data files at hand, and the icecube icetray software running on your machine. 
