import pandas as pd
import numpy as np
import argparse
import json
import os, os.path

##############################################################################################
#                                       get_retired.py
##############################################################################################
# Purpose: takes in composite classification/subject sets from name that neutrino website, 
# artificially creates dataset as it would be at the time that all subjects were retired for
# a given limit
# Usage: python get_retired.py <retirement_lim> <input_dir> <output_dir>
##############################################################################################

def getRetired(data_exports_dir, output_dir, retirement_lim):

   
    lim = retirement_lim                                                             #extract retirement limit desired
    input_dir = os.path.join(os.getcwd(), data_exports_dir)                                  #extract input directory (where to get ntn data exports)
    output_dir = os.path.join(os.getcwd(), output_dir)                                  #extract output directory (where to put modified ntn datasets)

    ''' Read in data '''
    classif = pd.read_csv(os.path.join(input_dir, 'name-that-neutrino-classifications.csv')) #read in classification set
    subj = pd.read_csv(os.path.join(input_dir, 'name-that-neutrino-subjects.csv'))           #read in subject set
    workflow = pd.read_csv(os.path.join(input_dir, 'name-that-neutrino-workflows.csv'))      #read in workflow set
    subj = subj[subj['workflow_id'] == 23715]                                            #slice for only desired workflow version

    ''' Create a hash map for subject ids'''
    subj_ids = np.array(subj['subject_id'])                                             #get array of subject ids
    subj_dict = {}                                                                      #create empty dict
    for id in subj_ids:
        subj_dict[id] = 0                                                               #initialize dict key to subj id, counts to zero

    subj_data = np.array(classif['subject_data'])                                       #get array of subject data dicts within classification dataframe
    metadata = json.loads(subj_data[0])                                                 #turn string into json dict

    ind = []                                                                            #initialize empty array to hold indices of desired classification entries
    n_subjects = 4273                                                                   #total number of subjects
    n_retired = 0
    retired = []                                                                        #initialize number of retired to 0

    
    for i in range(0, len(subj_data)):                                                  #loop over all the classifications until we count <lim> classifications for all 4273 subjects                        
        metadata = json.loads(subj_data[i])                                             #extra metadata
        key = int(list(metadata.keys())[0])                                             #get the subject id
        if key in subj_dict.keys():                                                     #make sure the id is in our list we made
            if (subj_dict[key] < lim):                                                  #execute if we haven't yet counted <lim> classifications for that subject
                subj_dict[key] += 1                                                     #increment counts by 1 for tht subject
                ind.append(i)                                                           #add the index of the classification in classif set to our list
            else:
                if (key not in retired):
                    n_retired += 1                                                      #otherwise, increment the count of retired subjects
                    retired.append(key)
            
            if (n_retired == n_subjects):                                               #break when we retire all the subjs
                break

    #replace classfications ount with new ret limit                                     #creating array same size as subj set of all ones, multiply by lim
    class_count = np.ones([len(subj), ], np.int8)*lim
    subj.iloc[:,6] = class_count                                                        #replace classification count row in subj set with our new list
    classif_new = classif.loc[ind]                                                      #slice initial classification set by the list of indices we just created

    '''Saving'''
    if (os.path.isdir(output_dir) == False):
        os.mkdir(output_dir)
    subj.to_csv(os.path.join(output_dir,'name-that-neutrino-subjects.csv'))
    classif_new.to_csv(os.path.join(output_dir, 'name-that-neutrino-classifications.csv'))
    workflow.to_csv(os.path.join(output_dir, 'name-that-neutrino-workflows.csv'))


if __name__ == '__main__':

    ''' Parsing command line args '''
    parser = argparse.ArgumentParser(
                    prog='get_retired',
                    description='Extracts classification data for a particular retirement limit for NtN dataset')
    parser.add_argument('retirement_lim', metavar='lim', type=int, nargs=1,
                    help='desired retirement limit')
    parser.add_argument('in_dir', metavar='indir', type=str, nargs=1, 
                    help='input directory')
    parser.add_argument('out_dir', metavar='outdir', type=str, nargs=1, 
                    help = 'output directory')

    args = parser.parse_args()
    lim = args.retirement_lim[0]                                                        #extract retirement limit desired
    input_dir = os.path.join(os.getcwd(), args.in_dir[0])                               #extract input directory (where to get ntn data exports)
    output_dir = os.path.join(os.getcwd(), args.out_dir[0])                             #extract output directory (where to put modified ntn datasets)

    getRetired(input_dir, output_dir, lim)                                              #run main func         