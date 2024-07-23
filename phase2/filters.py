import subprocess
import numpy as np
import sys
import os
import argparse
import csv
from tables import *
import pandas as pd
from matplotlib import pyplot as plt
import h5py
import glob
from icecube.icetray import I3Units
import icecube.MuonGun
from icecube import dataio, dataclasses, icetray, MuonGun
from I3Tray import *
from icecube.hdfwriter import I3HDFWriter
from APMCLabeler import APMCLabeler #import new custom MC labeler


'''
Author: Elizabeth Warrick, edited by Andrew Phillips
Date: 4/17/24
Purpose: Collection of filters for processing of raw simulated icecube data.
- Filter 1 MC labels all events.
- Filter 2 Randomly selects events based on a uniform energy distribution.
- Filter 3 Filters out non-DAQ frames.
'''

#gcd file location
gcd = '/home/aphillips/ntn/GeoCalibDetectorStatus_2012.56063_V1.i3.gz' #should probably make this an input (cmd line arg?) (AP)

#label dictionary (mapping from truth values, to english)
label_dict = {0:'unclassified',
                        1:'throughgoing_track',
                        2:'starting_track',
                        3:'stopping_track',
                        4:'skimming_track',
                        5:'contained_track',
                        6:'contained_em_hadr_cascade',
                        7:'contained_hadron_cascade',
                        8:'uncontained_cascade',
                        9:'glashow_starting_track',
                        10:'glashow_electron',
                        11:'glashow_tau_double_bang',
                        12:'glashow_tau_lollipop',
                        13:'glashow_hadronic',
                        14:'throughgoing_tau',
                        15:'skimming_tau',
                        16:'double_bang',
                        17:'lollipop',
                        18:'inverted_lollipop',
                        19:'throughgoing_bundle',
                        20:'stopping_bundle',
                        21:'tau_to_mu'}


"""
Filter 1:
- Use in ice split only. 
- Add mc_labeler tray.
- HDFWriter to make hdf table of info.
This is definetly not the most efficient since it needs to run the mc labeler on all frames, but it is what it is for now ¯\_(ツ)_/¯
"""
def label_events(infile, outdir):
    drive, ipath =os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    tray = I3Tray()
    tray.Add('I3Reader', FilenameList=[infile])
    tray.Add(APMCLabeler)
    tray.Add('I3Writer', 'EventWriter',
    FileName= outdir+'/mc_labeled_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Physics,
        icetray.I3Frame.Stream('S')],
        DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddSegment(I3HDFWriter, Output = f'{outdir}mc_labeled_{infile_name}.hd5', Keys = ['I3EventHeader','I3MCWeightDict',\
    'ml_suite_classification','NuGPrimary','PoleMuonLinefit', 'PoleMuonLinefitParams', 'PoleMuonLlhFitMuE', 'PoleMuonLlhFitFitParams',\
    'PoleMuonLlhFit','PolyplopiaInfo','PolyplopiaPrimary','I3MCTree','I3MCTree_preMuonProp','truth_classification', 'signal_charge', 
    'bg_charge', 'qtot'], SubEventStreams=['InIceSplit'])
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()
    
"""
Filter 2:
- Pick out random events that have a uniform energy distribution.
- Cutoff at PeV. 
- Select for only truths 1, 2, 3, 4, 6, 7, 8 (see enums.py)
- Takes a random seed (will want to program this to be optional later on.)
- Is standard to use bin num = 25 to follow most IceCube energy histograms
"""


#AP: editing out histogram stuff because unnecessary
def uniformenergy_events(hdf,bin_num, size,random_seed,subrun = False):
    #open hdf of desired i3 file
    hdf = f'{hdf}'
    hdf_file = h5py.File(hdf, "r+")

    #turn hdf into pandas dataframe.  

    event_id = hdf_file['I3EventHeader']['Event'][:] #Event ID
    run_id = hdf_file['I3EventHeader']['Run'][:] #Run ID

    #classifier predictions
    pred_skim_val = hdf_file['ml_suite_classification']['prediction_0000'][:] #dnn skim prediction
    pred_cascade_val = hdf_file['ml_suite_classification']['prediction_0001'][:] #dnn cascade prediction
    pred_tgtrack_val = hdf_file['ml_suite_classification']['prediction_0002'][:] #dnn tg track prediction
    pred_starttrack_val = hdf_file['ml_suite_classification']['prediction_0003'][:] #dnn start track prediction
    pred_stoptrack_val = hdf_file['ml_suite_classification']['prediction_0004'][:] #dnn stop track prediction
    truth_label = hdf_file['truth_classification']['value'][:] #label from mc_labeler (as a number)
    energy_val = hdf_file['NuGPrimary']['energy'][:] #energy
    zenith_val = hdf_file['NuGPrimary']['zenith'][:] #zenith
    ow = hdf_file['I3MCWeightDict']['OneWeight'][:] #oneweight

    #turn the above arrays into a pandas dataframe
    df = pd.DataFrame(dict(run = run_id, event = event_id, truth_classification = truth_label, pred_skim = pred_skim_val, pred_cascade = pred_cascade_val,\
        pred_tgtrack = pred_tgtrack_val, pred_starttrack = pred_starttrack_val, pred_stoptrack = pred_stoptrack_val,energy = energy_val, zenith = zenith_val,\
        oneweight = ow))
    
    hdf_file.close() #close hdf file
    
    #make array of english truth classification labels
    word_truth_labels = []
    for x in df['truth_classification']:
        word_truth_labels.append(label_dict[int(x)])
    label_str = f'#truth_classification_label'
    df[label_str] = word_truth_labels

    #add in other key-value pairs into dataframe

    #maximum category score returned by DNN
    df['max_score_val'] = df[['pred_skim','pred_cascade','pred_tgtrack','pred_starttrack','pred_stoptrack']].max(axis='columns')

    #based on max score, get dnn category
    df['idx_max_score'] = df[['pred_skim','pred_cascade','pred_tgtrack','pred_starttrack','pred_stoptrack']].idxmax(axis='columns')
    
    #Pick events "manually", cut at PeV-ish.
    truth_vals_list = [1, 2, 3, 4, 6, 7, 8] #these are the truth values we want to keep, corresponding to stop,start,tg,casc, and skim

    #take only rows with truth vals in above list
    df_masked = df[df['truth_classification'].isin(truth_vals_list)]

    #remove events with energies > PEV
    df_filtered = df_masked.loc[(np.log10(df_masked['energy'][:]) <= 6)]
    
    hist_df, bin_edges_df = np.histogram(np.log10(df_filtered['energy'][:]), density=False,bins=bin_num)
    #Use np.digitize to return index of the bin each energy value belongs to. 
    energy_bin_index = np.digitize(np.log10(df_filtered['energy']),bin_edges_df)
    #take any value less than fist bin is put in bin 1, and any value greater than 25th bin now in bin 25. 
    energy_bin_index = np.clip(energy_bin_index,1,bin_num) 
    #get bins and labels. 
    bins = bin_edges_df
    labels = np.arange(1,bin_num+1)
    #AP: I don't know what this is for, might be able to throw out.
    df_filtered['binned_log10E'] = pd.cut(x = np.log10(df_filtered['energy']), bins = bins, labels = labels, include_lowest = True) 
    #Loop through each bin number and mask out any events not in that bin number.

    seed=random_seed
    rng = np.random.default_rng(seed)

    random_event_indices = np.array([]) #empty array for event indices

    for i in np.arange(1,bin_num+1):
        e_bin = df_filtered.loc[df_filtered['binned_log10E']== i]
        random_energy_index = rng.choice(e_bin.index,replace=False, size=size) #pick specified number of events from each energy bin. 
        random_event_indices = np.append(random_event_indices,random_energy_index)
    
    
    random_event_energies = df_filtered.loc[df_filtered.index.intersection(random_event_indices)]

    height_df_idx_max_score = random_event_energies['idx_max_score'].value_counts(sort=False) #for making histograms of most confident ML prediction.
    
    if subrun == True:
        print("Please input subrun number:")
        sub_id = input()
        csv_name = f'random_events_uniform_energy_distrib_{run_id[0]}_{sub_id}.csv'
    else:
        csv_name = f'random_events_uniform_energy_distrib_{run_id[0]}.csv'
    print(csv_name)
    return random_event_energies.to_csv(csv_name)
    #return event_characterization_plots('random_events_uniform_energy_distrib.csv')

    
    
def events_cut(frame, event_csv):
    df = pd.read_csv(f'{event_csv}')
    uniform_events  = df['event'][:].values

    if frame['I3EventHeader'].sub_event_stream == 'NullSplit':
        return False
    elif frame['I3EventHeader'].sub_event_stream == 'InIceSplit':
        if frame['I3EventHeader'].event_id in uniform_events:
            return True
        else:
            return False

def filter2(infile, outdir,event_csv):
    drive, ipath =os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    tray = I3Tray()
    tray.Add('I3Reader', FilenameList=[infile])
    tray.AddModule(events_cut,event_csv = event_csv)
    tray.Add('I3Writer', 'EventWriter',
    FileName= outdir+'/uniform_energy_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Physics, 
        icetray.I3Frame.Stream('S')],
        DropOrphanStreams=[icetray.I3Frame.DAQ]) 
    tray.AddSegment(I3HDFWriter, Output = f'{outdir}uniform_energy_{infile_name}.hd5', Keys = ['I3EventHeader','I3MCWeightDict',\
    'ml_suite_classification','NuGPrimary','PoleMuonLinefit', 'PoleMuonLinefitParams', 'PoleMuonLlhFitMuE', 'PoleMuonLlhFitFitParams',\
    'PoleMuonLlhFit','PolyplopiaInfo','PolyplopiaPrimary','I3MCTree','I3MCTree_preMuonProp','classification'], SubEventStreams=['InIceSplit'])
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()

    
"""
Filter 3:
- Return Q frames only. 
- Split i3 file to be more managable for steamshovel to go through. 
- Maybe make another hdf just in case?
"""

#extracts daq frames, splits i3s into 2 mb files to ease steamhovel processing
def extract_daq(infile, run_id,ddir):
    drive, ipath =os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    name_run = f'daq_{run_id}'
    #print(name_run)
    #new_daq_out = os.join(outdir,name_run)
    #os.mkdir(f'daq_{run_id}')
    #os.mkdir('/home/aphillips/name-that-neutrino/output/daq_{}'.format(run_id))
    outdir = os.path.join(ddir,name_run,"")
    tray = I3Tray()
    tray.Add('I3Reader', FilenameList=[infile])
    tray.Add('I3Writer', 'EventWriter', #AP edit
    FileName= outdir+'daq_only-%04u_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Stream('S')],
        SizeLimit = 2*10**6,)
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()

