import numpy as np # type: ignore
import sys
import os, fnmatch
import pandas as pd # type: ignore
import glob
import json
from icecube import dataio, dataclasses, icetray, MuonGun # type: ignore
from I3Tray import * # type: ignore
from icecube.hdfwriter import I3HDFWriter # type: ignore
import h5py # type: ignore
from APMCLabeler import APMCLabeler
from phase2_filters import *

'''
Author: Andrew P
Date: 7/30/24
Purpose: Get phase2 i3s, perform cuts
'''


OUTDIR = '/scratch/aphillips/phase2_data/' #change to desired
NUMU_DIR = '/data/sim/IceCube/2020/filtered/test/newprocessing/neutrino-generator/21971/0000000-0000999/classifier/'
NUE_DIR = '/data/sim/IceCube/2020/filtered/test/newprocessing/neutrino-generator/22067/0000000-0000999/classifier/'

N = 8000 #number of events desired


if __name__ == '__main__':
    
    numu_subruns = pd.read_csv('/home/aphillips/name-that-neutrino/phase2/numu_files.csv')['subrun']
    nue_subruns = pd.read_csv('/home/aphillips/name-that-neutrino/phase2/nue_files.csv')['subrun']
    

    #counters for event types [n_skim, n_casc, n_tg, n_stop, n_start]
    event_counter = [0, 0, 0, 0, 0]
    categories = [0,1,2,3,4]
    

    print('Processing NuE files...')
    for subrun in nue_subruns:
        subrun = f'{subrun}'.rjust(6, '0')
        filename = f'classifier_rehyd_DST_IC86.2020_NuE.022067.{subrun}.i3.zst'
        filepath = os.path.join(NUE_DIR, filename)
        
        print(f'Processing file {filename} ...\n')
        
        label_events(filepath, OUTDIR) #mc labels all events
        make_csv(os.path.join(OUTDIR,f'mc_labeled_{filename}.hd5'), OUTDIR, subrun) #makes dataframe of desired events
        do_cuts(os.path.join(OUTDIR,f'mc_labeled_{filename}'), OUTDIR, os.path.join(OUTDIR, f'events_df_{subrun}.csv')) #cut on desired frames
        extract_daq(os.path.join(OUTDIR, f'cuts_mc_labeled_{filename}'), f'22067{subrun}', os.path.join(OUTDIR, 'nue_event_csvs')) #extract daq only and split into 2 mb sizes

        event_csv = pd.read_csv(os.path.join(OUTDIR, f'events_df_{subrun}.csv')) #open the csv we just created

        #loop over the categories
        for i in categories:
            event_counter[i] += len(event_csv[event_csv['ntn_category'] == i]) #count the number of events in each category

        categories = [i for i in categories if event_counter[i] <= N/5]

        print(f'[n_skim, n_cascade, n_throughgoing, n_stopping, n_starting = {event_counter}]')

        print(event_counter)
    '''
    print('Processing NuMu files...')
    for subrun in numu_subruns:
        subrun = f'{subrun}'.rjust(6, '0')
        filename = f'classifier_rehyd_DST_IC86.2020_NuMu.021971.{subrun}.i3.zst'
        filepath = os.path.join(NUMU_DIR, filename)
        
        print(f'Processing file {filename} ...\n')
        
        label_events(filepath, OUTDIR) #mc labels all events
        make_csv(os.path.join(OUTDIR,f'mc_labeled_{filename}.hd5'), OUTDIR, subrun) #makes dataframe of desired events
        do_cuts(os.path.join(OUTDIR,f'mc_labeled_{filename}'), OUTDIR, os.path.join(OUTDIR, f'events_df_{subrun}.csv'))
        extract_daq(os.path.join(OUTDIR, f'cuts_mc_labeled_{filename}'), f'21971{subrun}', os.path.join(OUTDIR, 'numu_event_csvs')) 
    '''