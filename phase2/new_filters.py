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
Author: Andrew Phillips
Date: 7/23/24

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
                        19:'throughgoing_track', #usually through-going bundle, but mapping to ntn categories
                        20:'stopping_track', #usually stopping_bundle, but mapping to ntn categs
                        21:'tau_to_mu'}

def make_csv(hdf):
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
    signal_charge = hdf_file['signal_charge']['value'][:] #signal charge
    bg_charge = hdf_file['bg_charge']['value'][:] #bg charge
    qtot = hdf_file['qtot']['value'][:] #total charge (bg +sig + noise)
    energy_val = hdf_file['NuGPrimary']['energy'][:] #energy
    zenith_val = hdf_file['NuGPrimary']['zenith'][:] #zenith
    ow = hdf_file['I3MCWeightDict']['OneWeight'][:] #oneweight

    
    #turn the above arrays into a pandas dataframe
    df = pd.DataFrame(dict(run = run_id, event = event_id, truth_classification = truth_label, pred_skim = pred_skim_val, pred_cascade = pred_cascade_val,\
        pred_tgtrack = pred_tgtrack_val, pred_starttrack = pred_starttrack_val, pred_stoptrack = pred_stoptrack_val,energy = energy_val, zenith = zenith_val,\
        oneweight = ow, signal_charge = signal_charge, bg_charge = bg_charge, qtot = qtot))
    
    df['qratio'] = np.divide(signal_charge, signal_charge+bg_charge) #charge ratio
    df['log10_max_charge'] = np.maximum(bg_charge, signal_charge)
    hdf_file.close()
    
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
    truth_vals_list = [1, 2, 3, 4, 6, 7, 8, 19, 20] #these are the truth values we want to keep, corresponding to stop,start,tg,casc, and skim

    #take only rows with truth vals in above list
    df_masked = df[df['truth_classification'].isin(truth_vals_list)]

    #remove events with energies > PEV
    df_filtered = df_masked.loc[(np.log10(df_masked['energy'][:]) <= 6)]
    #remove events with dnn confidence <= 0.5
    df_filtered = df_filtered[df_filtered['max_score_val'] <= 0.5]
    
    #remove events with 0.2 <= qratio <= 0.8
    df_filtered = df_filtered[(df_filtered['qratio'] <= 0.2) | (df_filtered['qratio'] <= 0.8)]
    
    #remove events with log10(max_charge) < 1
    df_filtered = df_filtered[df_filtered['log10_max_charge'] >= 1]
    
    return df_filtered.to_csv(os.path.join(os.getcwd(), 'events_df.csv'))



def cuts(frame, event_csv):
    
    df = pd.read_csv(f'{event_csv}')
    events  = df['event'][:].values
                              
    if frame['I3EventHeader'].sub_event_stream == 'InIceSplit':
                              
        event_id = frame['I3EventHeader'].event_id     
        if event_id in events:
            return True
        else:
            return False
                              
    else:
        return False
    
    
def do_cuts(infile, outdir,event_csv):
    drive, ipath =os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    tray = I3Tray()
    tray.Add('I3Reader', FilenameList=[infile])
    tray.AddModule(cuts,event_csv = event_csv)
    tray.Add('I3Writer', 'EventWriter',
    FileName= outdir+'/cuts_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Physics, 
        icetray.I3Frame.Stream('S')],
        DropOrphanStreams=[icetray.I3Frame.DAQ]) 
    tray.AddSegment(I3HDFWriter, Output = f'{outdir}cuts_{infile_name}.hd5', Keys = ['I3EventHeader','I3MCWeightDict',\
    'ml_suite_classification','NuGPrimary','PoleMuonLinefit', 'PoleMuonLinefitParams', 'PoleMuonLlhFitMuE', 'PoleMuonLlhFitFitParams',\
    'PoleMuonLlhFit','PolyplopiaInfo','PolyplopiaPrimary','I3MCTree','I3MCTree_preMuonProp','classification'], SubEventStreams=['InIceSplit'])
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()
                              

