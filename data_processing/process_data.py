import pandas as pd
pd.options.mode.chained_assignment = None #suppress this annoying warning from pandas
import numpy as np
import os, os.path
import json
import argparse

if __name__ == '__main__':

    ''' Parsing command line args '''
    parser = argparse.ArgumentParser(
                    prog='get_retired',
                    description='Extracts classification data for a particular retirement limit for NtN dataset')
    parser.add_argument('events_dir', metavar='evt_dir', type=str, nargs=1, 
                    help='location of events csv from cobalt')
    parser.add_argument('ntn_dir', metavar='ntn_dir', type=str, nargs=1, 
                    help = 'location of ntn exports')

    args = parser.parse_args()
    event_csvs = args.events_dir[0]                                                                                     #extract input directory (where to get csvs with subject set data)
    ntn_exports = args.ntn_dir[0]                                                                                       #where to put stuff later

    subject_set_ids = [112109, 112391, 112425, 112433, 112464, 112392, 112414, 112418, 112498, 
                       112473, 112487, 112492, 112467, 112116, 112481, 112501, 112118, 112119, 112120]                  #subject set ids we want
    

    '''Slice up subject data into subject sets'''
    ntn_subjects_all = pd.read_csv(os.path.join(ntn_exports, 'name-that-neutrino-subjects.csv'))                        #csv containing all subjects from zooniverse
    ntn_subjects_23715 = ntn_subjects_all[ntn_subjects_all['workflow_id'] == 23715]                                     #get only the workflow we want
    event_ids = []                                                                                                      #array of event ids
    dnn_choices = []
    energy = []
    run_ids = []

    #first loop over the rows and extract stuff from the metadata dict
    for idx in range(0, len(ntn_subjects_23715.index)):
        metadata = ntn_subjects_23715['metadata'][idx]                                                                  #get the metadata
        md_dict = json.loads(metadata)                                                                                  #turn string into dict
        event_ids.append(md_dict['event'])    
        energy.append(md_dict['energy'])                                                                                #add event id to list
        dnn_choices.append(md_dict['idx_max_score'])                                                                    #add dnn choice to list
        run_ids.append(md_dict['run'])                                                                                  #add run id to list

    ntn_subjects = ntn_subjects_23715[['subject_id', 'workflow_id', 'subject_set_id']]                                  #get only the columns we need
    ntn_subjects['event_id'] = event_ids                                                                                #add in event ids
    ntn_subjects['idx_max_score'] = dnn_choices                                                                         #add in dnn choices
    ntn_subjects['run'] = run_ids                                                                                       #add in run ids
    ntn_subjects['energy'] = energy


    for id in subject_set_ids:                                                                                          #loop over subject ids
        fname = os.path.join(os.getcwd(), 'subjects_filtered', 'ntn_subjects_{}.csv'.format(id))                        #construct filename
        df = ntn_subjects[ntn_subjects['subject_set_id']==id]                                                           #create new data frame
        df = df.drop_duplicates(subset=['event_id'])                                                                    #for some reason some of the csvs have duplicate rows so those must be dropped!
        df.to_csv(fname, index=False)                                                                                   #save as csv


    '''Extract only NtN events, because not all make it through steamshovel'''
    files = os.listdir(os.path.join(os.getcwd(), event_csvs))
    #print(len(files))
    for idx in range(0, len(files)):
        path = os.path.join(os.getcwd(), 'event_csvs', files[idx])                                                      #construct filepath
        df = pd.read_csv(path)                                                                                          #open the csv as a pandas dataframe
        if idx==0:                                                                                                      #if first loop create a new empty dataframe with the same columns
            df2 = pd.DataFrame(columns = df.columns)

        size = len(df.index)                                                                                            #get the size of the current csv
        ssid_array = np.ones((size, 1), dtype=np.int8)*int(subject_set_ids[idx])                                        #create a column containing the subject set id
        df['Subject Set ID'] = ssid_array                                                                               #set new column equal to subject set ids
        eventIds = np.array(df['event'])                                                                                #get event ids
        fname = './subjects_filtered/ntn_subjects_{}.csv'.format(subject_set_ids[idx])                                  #construct file name with subjects in curretn subject set
        subjects = pd.read_csv(fname)                                                                                   #read that csv
        subjects = subjects.sort_values(by=['event_id'], axis=0)                                                        #make sure it's sorted
        eventIds2 = np.array(subjects['event_id'])                                                                      #create second array with event ids from subject set file
        dnn = np.array(subjects['idx_max_score'])                                                                       #get dnn choices from subject set
        subject_id = np.array(subjects['subject_id'])                                                                   #get subject ids
        run_id = np.array(subjects['run'])                                                                              #get run ids
        energy = np.array(subjects['energy'])
        ind = np.nonzero(np.in1d(eventIds, eventIds2))[0]                                                               #find intersection between the two arrays of event ids
        df = df.iloc[ind]                                                                                               #trim event data down to only those that passed into name that neutrino
        df['DNN Classification'] = dnn                                                                                  #create new column for dnn choices
        df['Subject ID'] = subject_id                                                                                   #create new column for subject ids 
        df['Energy'] = energy
        df2 = pd.concat([df2, df])                                                                                      #concatenate new df to main dataframe                                            
    df2 = df2.drop('Unnamed: 0', axis=1)                                                                                #get rid of this weird ghost column
    df2.to_csv('test.csv')
    '''Add in user data'''
    user_data = pd.read_csv(os.path.join(ntn_exports,'consensus_reduced.csv'))                                           #get user classification data
 
    for i in range(0, len(user_data)):                                                                                   #get rid of duplicate events in user data
        if user_data['subject_id'][i] not in list(df2['Subject ID']):
            user_data = user_data.drop(i)
    
    df2 = df2.sort_values(by=['Subject ID'], axis=0)                                                                    #now this guy should line up 1-1 with user data.
    user_choices = user_data['data.most_likely']                                                                        #get user choices column 

    user_agreement = user_data['data.agreement']                                                                        #get user agreement column
    df2['User Consensus Classification'] = np.array(user_choices)                                                       #add user classifications to dataframe
    df2['User Agreement'] = np.array(user_agreement)                                                                    #add user agreement to dataframe

    df2 = df2[['Subject ID', 'run', 'Subject Set ID', 'event', 'Energy', 'edep', 'bg_charge', 
               'coinc_muons', 'qtot', 'radialcog', 'verticalcog', 'peRatio', 'truth_classification', 
               'corsika_label', 'DNN Classification', 'User Consensus Classification']]                                 #reorder columns
    
    df2.to_csv(os.path.join(os.getcwd(), 'ntn_data_processed.csv'))                                                     #save                                 

                                                             
 