import numpy as np
import sys
import os, fnmatch
import pandas as pd
import glob
from icecube import dataio, dataclasses, icetray, MuonGun
from I3Tray import *
from icecube.hdfwriter import I3HDFWriter
from mc_labeler import MCLabeler
from ap_modules import CorsikaLabeler, QTot
import h5py


'''Mapping for event categories'''
CLASS_MAPPING = {
    0:'unclassified',
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
    21:'tau_to_mu'
}

'''Function to apply custom modules to i3 files'''

def apply_modules(input_file, output_dir):
    infile =  input_file
    outdir = output_dir
    drive, ipath = os.path.splitdrive(infile)
    path, ifn = os.path.split(ipath)
    infile_name = infile.split('/')[-1]
    tray = I3Tray()
    outfile = outdir+'/ap_modules'+infile_name
    hdf_name = f'{outdir}/ap_modules_{infile_name}.hd5'
    tray.Add('I3Reader', FilenameList=[infile])
    tray.Add(MCLabeler)
    tray.Add(CorsikaLabeler)
    tray.AddModule("I3NullSplitter", "fullevent")
    tray.Add(QTot, 'Qtotal', Where = 'Qtot')
    tray.Add('I3Writer', 'EventWriter', ##dont need to create new i3
    FileName= outdir+'/ap_modules_'+infile_name,
        Streams=[icetray.I3Frame.TrayInfo,
        icetray.I3Frame.Geometry,
        icetray.I3Frame.Calibration,
        icetray.I3Frame.DetectorStatus,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Stream('S')])
        #DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddSegment(I3HDFWriter, Output = hdf_name, Keys = ['I3EventHeader',\
    'classification', 'corsika_label', 'Qtot',  \
    'coincident_muons', 'bg_charge', 'subject_id'], SubEventStreams=['fullevent'])
    tray.AddModule('TrashCan','can')

    tray.Execute()
    tray.Finish()
    return (outfile, hdf_name)

'''Extract events and put info into csvs'''
def process_data(hdf): #, out_dir, subject_set_id):
    #open hdf of desired i3 file
    hdf = f'{hdf}'
    hdf_file = h5py.File(hdf, "r+")
    event_id = hdf_file['I3EventHeader']['Event'][:] #Event ID
    run_id = hdf_file['I3EventHeader']['Run'][:] #Run ID
    truth_label = hdf_file['classification']['value'][:] #mc_labeler value
    cr_label = hdf_file['corsika_label']['value'][:] #mc_labeler value
    subj_id = hdf_file['subject_id']['value'][:]
    bg_charge = hdf_file['bg_charge']['value'][:]
    Qtot = hdf_file['Qtot']['value'][:]
    coincident_muons = hdf_file['coincident_muons']['value'][:]
    #Dataframe time
    df = pd.DataFrame(dict(run = run_id, event = event_id, subject_id = subj_id, truth_classification = truth_label, \
    corsika_label = cr_label, bg_charge = bg_charge, qtot = Qtot, \
     coinc_muons = coincident_muons))
    hdf_file.close() #close hdf file now that dataframe is made. 
    #csv_name = f'{out_dir}/all_evt_features_{run_id[0]}_{subject_set_id}.csv'
    #df.to_csv(csv_name)
    return df #csv_name
    