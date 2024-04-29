import pandas as pd # type: ignore
import numpy as np # type: ignore
import argparse

'''
filename: consolidate_data.py
author: andrew phillips
purpose: combines user,dnn, and mc datasets into single dataframe
'''

def consolidate(path_to_ntn, path_to_mc, outfile_name):

    mc_data = pd.read_csv(path_to_mc)                          #Monte-carlo truth labels, plus custom data properties, e.g., corsika labels, qtot, etc.
    ntn_data = pd.read_csv(path_to_ntn)                          #Processed name that neutrino data
    mc_data = mc_data.sort_values(by=['subject_id'])            #sort both dataframes by subject id
    ntn_data = ntn_data.sort_values(by=['subject_id'])

    #we must employ a little trick to get rid of duplicate events
    #I do this by making a list of (event_id, subject_set_id) values pairs for every event, pushing it to the dataframe, and deleting duplicate rows
    ssids = ntn_data['subject_set_id']
    event_id = ntn_data['event_id']
    tmp = [obj for obj in zip(ssids, event_id)]
    ntn_data.insert(0, 'tmp', tmp)
    ntn_data = ntn_data.drop_duplicates(subset=['tmp'])

    corsika_labels = list(mc_data['corsika_label']) #corsika labels from corsika labeler
    bg_charge = list(mc_data['bg_charge'])          #background charge contribution
    signal_charge = list(mc_data['signal_charge']) 
    qtot = list(mc_data['qtot'])                    #total charge in event
    qratio = np.divide(bg_charge, qtot)             #charge ratio

    #insert all the new stuff 
    ntn_data.insert(0, 'corsika_label', corsika_labels)
    ntn_data = ntn_data.replace({'corsika_label':[0, 1, 3, 4, 19, 20 ]},{'corsika_label':[5,2,4,0,2,4 ]},regex=False)
    ntn_data.insert(0, 'bg_charge', bg_charge)
    ntn_data.insert(0, 'qtot', qtot)
    ntn_data.insert(0, 'qratio', qratio)
    ntn_data.insert(0, 'signal_charge', signal_charge)

    #creating a new dataframe with only the relevant columns, save as csv
    ntn_consolidated = ntn_data[['subject_set_id', 'subject_id', 'event_id','qtot', 'bg_charge', 'signal_charge', 'qratio', 'corsika_label' , 'data.most_likely', 'idx_max_score', 'truth_classification']]
    ntn_consolidated = ntn_consolidated.rename(columns={'data.most_likely':'user_classification', 'idx_max_score':'dnn_classification'})
    ntn_consolidated.to_csv(outfile_name, index=False)

if __name__ == '__main__':

    ''' Parsing command line args '''
    parser = argparse.ArgumentParser(
                    prog='consolidate_data.py',
                    description='Combines user, dnn, and mc into a single dataframe')
    parser.add_argument('path_to_ntn', metavar='path_to_ntn', type=str, nargs=1, 
                    help='location of user/dnn dataframe')
    parser.add_argument('path_to_mc', metavar='path_to_mc', type=str, nargs=1, 
                    help = 'location of mc dataframe')
    parser.add_argument('outfile_name', metavar='outfile_nme', type=str, nargs=1, 
                    help = 'name of output file')

    args = parser.parse_args()
    path_to_ntn = args.path_to_ntn[0]                                                                      #extract input directory (where to get csvs with subject set data)
    
    path_to_mc = args.path_to_mc[0]                                                                        #where to put stuff later
    outfile_name = args.outfile_name[0]                                                                    #where to put stuff later
    consolidate(path_to_ntn, path_to_mc, outfile_name)