
import pandas as pd
import os
import sys
import json #reading java strings into python dictionaries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

##############################################################################################
#                                phase1_data_analysis.py
##############################################################################################
# Purpose: perform analysis on aggregated NtN data
# Usage: python phase1_data_analysis <lim> <indir> <outdir>
##############################################################################################


''' Function to generate confusion matrix labels '''

def getUncertaintyLabels(matrix_df):
    labels = []
    nrows, ncols = matrix_df.shape
    for i in range(0,nrows):
        col_sum = matrix_df.iloc[:,i].sum()
        for k in range(0,ncols):
            val = matrix_df.iloc[k,i]
            n_val = (val/col_sum.astype(float))*100
            val_percent_round = "%.1f" % n_val
            lab_n = '%.1f%%'%n_val
            lab_v = '%d/%d'%(val,col_sum)
            lab='%.1f\%%\n%d/%d' % (n_val, val, col_sum)
            labelc= str(lab_n)+"\n"+str(lab_v)
            label = f"{str(lab_n)} {str(lab_v)}"
            labels.append(str(lab))
    labels_array = np.asarray(labels)
    labels_val = labels_array.reshape(5,5)
    labels_new = labels_val.T
    return labels_new

def consolidateData(data_consensus, ntn_subjects, retirement_lim, outdir):

    #initiate empty lists of keys wanted. 
    runs = []
    events = []
    energies = []
    zeniths = []
    oneweights = []
    pred_skims = []
    pred_cascades = []
    pred_tgtracks = []
    pred_stoptracks = []
    pred_starttracks = []
    binned_log10Es = []
    idx_max_scores = []
    max_score_vals = []
    pred_stoptracks = []
    truth_classifications = []
    subj_ids = []
    total_class_counts = []

    #go through the classifications csv
    for i in range(len(ntn_subjects)):
        dict1 = json.loads(ntn_subjects['subject_data'][i]) #each event/subject id is its own dictionary in classifications csv. 
        for key in dict1.keys(): #get the keys for each event's dictionary
            if key in subj_ids: #if you already have the event saved, go to the next event. 
                continue
            subj_ids.append(key)  #add the event (key) to a list
            total_class_counts.append(retirement_lim) #cheater version (AP), they're all retired so we don't really have to check, by virtue of the way we created it
            runs.append(dict1[key]['run']) #get the other desired keys and save as a list in order of appearence. 
            events.append(dict1[key]['event'])
            energies.append(dict1[key]['energy'])
            zeniths.append(dict1[key]['zenith'])
            oneweights.append(dict1[key]['oneweight'])
            pred_skims.append(dict1[key]['pred_skim'])
            pred_cascades.append(dict1[key]['pred_cascade'])
            pred_tgtracks.append(dict1[key]['pred_tgtrack'])
            pred_stoptracks.append(dict1[key]['pred_stoptrack'])
            pred_starttracks.append(dict1[key]['pred_starttrack'])
            binned_log10Es.append(dict1[key]['binned_log10E'])
            idx_max_scores.append(dict1[key]['idx_max_score'])
            max_score_vals.append(dict1[key]['max_score_val'])
            truth_classifications.append(dict1[key]['truth_classification'])

    data = {'subject_id':subj_ids, 'total_class_count':total_class_counts,'run':runs,'event':events,'energy':energies,
        'zenith':zeniths,'oneweight':oneweights,'pred_skim':pred_skims,'pred_cascade':pred_cascades,
        'pred_tgtrack':pred_tgtracks,'pred_starttracks':pred_starttracks,'pred_stoptrack':pred_stoptracks,
        'binned_log10E':binned_log10Es,'idx_max_score':idx_max_scores,'max_score_val':max_score_vals,'truth_classification':truth_classifications
    }   

    icecube_info = pd.DataFrame(data)
    icecube_info_sorted = icecube_info.sort_values('subject_id')
    icecube_info_sorted = icecube_info_sorted.reset_index()
    icecube_info_sorted = icecube_info_sorted.drop(['index'],axis=1)
    #Merge IceCube and User dataframes. 
    result_consensus = pd.concat([data_consensus, icecube_info_sorted], axis=1) 
    result_consensus = result_consensus.replace({'truth_classification':[1,2,3,4,5,6,7,8]},{'truth_classification':[2,3,4,0,4,1,1,0]},regex=False)
    #Change how user labels appear
    result_consensus = result_consensus.replace({'data.most_likely':['Skimming Track','Cascade','Through-Going Track','Starting Track','Stopping Track']},{'data.most_likely':[0,1,2,3,4]},regex=False)
    #Change how ML labels appear
    result_consensus = result_consensus.replace({'idx_max_score':['pred_skim','pred_cascade','pred_tgtrack','pred_starttrack','pred_stoptrack']},{'idx_max_score':[0,1,2,3,4]},regex=False)
    
    csv_name = os.path.join(outdir, 'ntn-result-consensus.csv')
    result_consensus.to_csv(csv_name)
    return csv_name


def makePlots(result_consensus, retirement_lim, outdir):

    '''Result retired'''
    result_retired = result_consensus[(result_consensus['total_class_count']>retirement_lim-1)]
    result_retired_user55 = result_consensus[(result_consensus['total_class_count']>retirement_lim-1) & (result_consensus['data.agreement']>=0.55)]
    types = ['Skimming','Cascade','Through-Going\nTrack','Starting\nTrack','Stopping\nTrack']

    if (os.path.isdir(os.path.join(os.getcwd(), outdir, 'plots')) == False): #create directory for plots, if not existent
        os.mkdir(os.path.join(os.getcwd(), outdir, 'plots'))

    '''Make bin labels, digitize data'''
    #Get the possible value of the consensus voter fractions
    x = result_retired['data.agreement'].value_counts(ascending=False).keys().tolist() #use for tick marks. 
    xtick_labels = []
    min_val = retirement_lim/5
    bin_edges = np.histogram_bin_edges(result_retired['data.agreement'][:],bins=int(retirement_lim*4/5)+1, range= ((min_val-0.5)/retirement_lim,(retirement_lim+0.5)/retirement_lim))
    #bin_edges = np.histogram_bin_edges(result_retired['data.agreement'][:],bins=13, range= (1/6,))
    for i in np.arange(0,len(bin_edges)):
        xtick_labels.append("{:0.2f}".format(bin_edges[i]))
        #Need to specify bin edges so that the scores are centered in the bins
    #min_val = retirement_lim/5
    np.digitize(result_retired['data.agreement'], bin_edges, right=True)
    new_bin_labels = np.arange((min_val-1)/retirement_lim,1+1/retirement_lim,1/retirement_lim).round(2)
    

    ########################################## HISTOGRAM OF MAX USER SCORES ########################################################
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 20
    })
    plt.figure()
    result_retired['data.agreement'].plot(kind='hist',bins=bin_edges,logy=False,edgecolor='black',figsize=(12,7))
    plt.xlabel(f'Consensus User Vote Fraction (Retirement Limit = {retirement_lim})',fontsize=25,labelpad=15)
    plt.ylabel('Number of Events',fontsize=25,labelpad=15)
    plt.xticks(new_bin_labels,labels=new_bin_labels,rotation=45,fontsize=15)
    plt.xlim(0.15,1.05)
    plt.ylim(0,1000)
    #plt.title('Distribution of Max User Score')
    #plt.savefig('{}/plots/user_max_score.png'.format(outdir))
    plt.savefig(os.path.join(outdir, 'plots', 'user_max_score.png'), bbox_inches='tight')
    #plt.show()

     ######################################## HISTOGRAM OF MAX USER SCORES , FREQUENCY PLOTS ##########################################
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 20
    })

    plt.figure(figsize=(12,7))
    counts, bins = np.histogram(result_retired['data.agreement'], bins=bin_edges)
    plt.stairs(counts/sum(counts), bins, fill=True, edgecolor='black', linewidth=1.2)
    plt.xlabel(f'Consensus User Vote Fraction (Retirement Limit = {retirement_lim})',fontsize=25,labelpad=15)
    plt.ylabel(r'Frequency',fontsize=25,labelpad=15)
    plt.xticks(new_bin_labels,labels=new_bin_labels,rotation=45,fontsize=15)
    plt.xlim(0.15,1.05)
    plt.savefig(os.path.join(outdir, 'plots', 'user_max_score_freq.png'), bbox_inches='tight')
    plt.close()
    #plt.show()

     ######################################## HISTOGRAM OF MAX DNN SCORES ###############################################################
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size":20
    })
    plt.figure()
    result_retired['max_score_val'].plot(kind='hist',bins=bin_edges,logy=False,edgecolor='black',figsize=(12,7))
    plt.xlabel(r'DNNClassifier Score',fontsize=25,labelpad=15)
    plt.ylabel('Number of Events',fontsize=25,labelpad=15)
   
    plt.xticks(new_bin_labels,labels=new_bin_labels,rotation=45,fontsize=15)
    plt.xlim(0.15,1.05)
    plt.savefig(os.path.join(outdir, 'plots', 'dnn_max_score.png'), bbox_inches='tight')
    plt.close()
    #plt.show()
    ############################################# DNN v MC CONFUSION #####################################################################
    confusion_matrix_ml_truth_norm = pd.crosstab(result_retired['idx_max_score'], result_retired['truth_classification'], rownames=['DNN Max Category'], colnames=['MC Truth Label'], margins=False,normalize='columns')
    #confusion_matrix_user_ml_55_norm_20ret = confusion_matrix_user_ml_55_norm
    confusion_matrix_ml_truth = pd.crosstab(result_retired['idx_max_score'], result_retired['truth_classification'], rownames=['DNN Max Category'], colnames=['MC Truth Label'], margins=False)
    fig, ax = plt.subplots(figsize=(13,13))
    ax = sns.heatmap(confusion_matrix_ml_truth_norm, annot=getUncertaintyLabels(confusion_matrix_ml_truth), annot_kws={"size": 15}, fmt='',cmap='Blues',xticklabels=types,yticklabels=types,vmin=0.0,vmax=1.0,cbar_kws={'label':'Fraction'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel('Fraction',fontsize=16)
    #plt.show()
    plt.ylabel('DNN Max Category',fontsize=25,labelpad=15)
    plt.xlabel('MC Truth Label',fontsize=25,labelpad=15)
    plt.yticks(np.arange(5)+0.5,types,
            rotation=0, fontsize="15", va="center")
    plt.xticks(np.arange(5)+0.5,types,
            rotation=0, fontsize="15")
    #plt.savefig("ml_user_cm_cut.png")
    #plt.savefig('{}/plots/dnn_mc_confusion.png'.format(outdir))
    plt.savefig(os.path.join(outdir, 'plots', 'dnn_mc_confusion.png'), bbox_inches='tight')
    plt.close()
    #plt.show()

    ############################################## USER v MC CONFUSION ######################################################################
    confusion_matrix_user_truth_norm = pd.crosstab(result_retired['data.most_likely'], result_retired['truth_classification'], rownames=['User Max Category'], colnames=['MC Truth Label'], margins=False,normalize='columns')
    #confusion_matrix_user_ml_55_norm_20ret = confusion_matrix_user_ml_55_norm
    confusion_matrix_user_truth = pd.crosstab(result_retired['data.most_likely'], result_retired['truth_classification'], rownames=['User Max Category'], colnames=['MC Truth Label'], margins=False)
    fig, ax = plt.subplots(figsize=(13,13))
    ax = sns.heatmap(confusion_matrix_user_truth_norm, annot=getUncertaintyLabels(confusion_matrix_user_truth), annot_kws={"size": 15}, fmt='',cmap='Blues',xticklabels=types,yticklabels=types,vmin=0.0,vmax=1.0,cbar_kws={'label':'Fraction'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel('Fraction',fontsize=16)
    #plt.show()
    plt.ylabel('User Max Category',fontsize=25,labelpad=15)
    plt.xlabel('MC Truth Label',fontsize=25,labelpad=15)
    plt.yticks(np.arange(5)+0.5,types,
            rotation=0, fontsize="15", va="center")
    plt.xticks(np.arange(5)+0.5,types,
            rotation=0, fontsize="15")
    #plt.savefig("ml_user_cm_cut.png")
    #plt.savefig('{}/plots/user_mc_confusion.png'.format(outdir))
    plt.savefig(os.path.join(outdir, 'plots', 'user_mc_confusion.png'), bbox_inches='tight')
    plt.close()
    #plt.show()


    #e.g., of the 280 times when DNN predicted cascade with .55 confidence or better, the user also predicted cascade with .55 or better confidence 257 times.

    ############################################################################################ USER v DNN CONFUSION ######################################################################################
    confusion_matrix_user_ml_norm = pd.crosstab(result_retired['data.most_likely'], result_retired['idx_max_score'], rownames=['User'], colnames=['ML'], margins=False,normalize='columns')
    #confusion_matrix_user_ml_norm_20ret = confusion_matrix_user_ml_55_norm
    confusion_matrix_user_ml = pd.crosstab(result_retired['data.most_likely'], result_retired['idx_max_score'], rownames=['User'], colnames=['ML'], margins=False)
    fig, ax = plt.subplots(figsize=(13,13))
    ax = sns.heatmap(confusion_matrix_user_ml_norm, annot=getUncertaintyLabels(confusion_matrix_user_ml), annot_kws={"size": 15}, fmt='',cmap='Blues',xticklabels=types,yticklabels=types,vmin=0.0,vmax=1.0,cbar_kws={'label':'Fraction'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel('Fraction',fontsize=16)
    #plt.show()
    plt.ylabel('User Max Category',fontsize=25,labelpad=15)
    plt.xlabel('DNN Max Category',fontsize=25,labelpad=15)
    plt.yticks(np.arange(5)+0.5,types,
            rotation=0, fontsize="15", va="center")
    plt.xticks(np.arange(5)+0.5,types,
            rotation=0, fontsize="15")
    #plt.savefig("ml_user_cm_cut.png")
    plt.savefig(os.path.join(outdir, 'plots', 'user_dnn_confusion.png'), bbox_inches='tight')
    plt.close()
    #plt.show()


    #e.g., of the 280 times when DNN predicted cascade with .55 confidence or better, the user also predicted cascade with .55 or better confidence 257 times.


if __name__ == '__main__':

    ''' Parsing command line args '''
    parser = argparse.ArgumentParser(
                    prog='phase1_data_analysis',
                    description='Data anlysis for phase 1 name that neutrino data')
    parser.add_argument('retirement_lim', metavar='lim', type=int, nargs='+',
                    help='desired retirement limit')
    parser.add_argument('in_dir', metavar='indir', type=str, nargs=1, 
                    help='input directory')
    parser.add_argument('out_dir', metavar='outdir', type=str, nargs=1, 
                    help = 'output directory')

    args = parser.parse_args()
    retirement_lim = args.retirement_lim[0]                                                                    #extract retirement limit desired
    input_dir = os.path.join(os.getcwd(), args.in_dir[0])                                                      #extract input directory (where to get ntn data exports)
    outdir = os.path.join(os.getcwd(), args.out_dir[0])                                                    #extract output directory (where to put modified ntn datasets)



    ''' '''
    #Pull out the columns from csv's made in step 1. 

    # the `data.*` columns are read in as strings instead of arrays
    data_consensus = pd.read_csv(os.path.join(input_dir, 'consensus_reduced.csv'))                             #read in reduced data
    ntn_subjects = pd.read_csv(os.path.join(input_dir, 'name-that-neutrino-classifications.csv'))              #read in classification data

    csv_name = consolidateData(data_consensus, ntn_subjects, retirement_lim, outdir)                                                   #consolidate 

    result_consensus = pd.read_csv(csv_name)
    makePlots(result_consensus, retirement_lim)                                                                #make plots

    