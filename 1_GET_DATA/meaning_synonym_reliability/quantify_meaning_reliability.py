"""
Script for loading the item-level synonym and meaning ratings for meaning and synonyms (for each participant) and quantifying
the reliability of the norms.

"""
from utils import *
from paths import *

def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Arguments for running reliability analyses')

    # Brain specific
    parser.add_argument('--DATADIR', default=join(DATADIR, '3_PREP_ANALYSES'),
                        type=str, help='Where to load the final experiment data from (used for analyses)')
    parser.add_argument('--EXPTFILENAME', default='exp1_data_with_norms_reordered_20221029.csv',
                        type=str, help='Name of the file to load the experiment data from (used for analyses)')
    parser.add_argument('--NORMDATADIR', default=EXPT1_NORMDIR,
                        type=str, help='Where to load the subject-level synonym/meaning data from (used for reliability)')
    parser.add_argument('--RELIABILITYRESULTDIR', default=RELIABILITYDIR,
                        type=str, help='Where to save the reliability results')
    parser.add_argument('--SUBJECTSPLITDIR', default=EXPT1_SUBJECT_SPLITDIR,
                        type=str, help='Where to save the generated subject splits')

    parser.add_argument('--expt', default=1, type=int, help='Which experiment to run the reliability analyses for')
    parser.add_argument('--norm', default='syn', type=str, help='Which norm to run the reliability analyses for. '
                                                                    'Options: meaning, syn')
    parser.add_argument('--n_it', default=1000, type=int, help='How many iterations (splits) to run the reliability analyses for')
    parser.add_argument('--save', default=True, type=str2bool, help='Whether to save the results')
    
    args = parser.parse_args(raw_args)
                        

    ###### LOAD DATA (USED FOR ANALYSES) #######
    # fname_data_exp1 = "../3_PREP_ANALYSES/exp1_data_with_norms_reordered_20221029.csv"
    df_exp = pd.read_csv(join(args.DATADIR, args.EXPTFILENAME), index_col=0)
    
    ###### LOAD DATA (USED FOR RELIABILITY) #######
    # In each folder that startswith 'list', we want to load a csv file with this name: list{}_meaningdata_subjlevel.csv
    list_folders = [f for f in os.listdir(args.NORMDATADIR) if f.startswith('list')]
    
    df_lst = []
    for folder in list_folders:
        # Load the csv file
        fname = f'list{folder[-1]}_{args.norm}data_subjlevel.csv'
        df = pd.read_csv(join(args.NORMDATADIR, folder, fname))
        df_lst.append(df)
    
    df_norms = pd.concat(df_lst, axis=0)
    
    # Drop Experiment = "non_words"
    df_norms = df_norms[df_norms['Experiment'] != 'non_words']
    
    unique_input = df_norms['Input.trial_'].unique()
    print(f'Number of unique inputs: {len(unique_input)}')
    
    # Only keep the Input.trial_ that are in df_exp column: word_upper
    df_norms = df_norms[df_norms['Input.trial_'].isin(df_exp['word_upper'])]

    # For expt2 (synonyms), we do not have Participant/WorkerID 'A32ADG69LXJU2R' in the compiled norms from Kyle/Ted (orig_exclusions), so let's drop this one participant
    # For expt2 (meanings), we do not have Participant/WorkerID 'A1CPKE6NZE67E4' in the compiled norms from Kyle/Ted (orig_exclusions), so let's drop this one participant
    if args.expt == 2 and args.norm == 'syn':
        df_norms = df_norms[df_norms['Participant'] != 'A32ADG69LXJU2R']
    if args.expt == 2 and args.norm == 'meaning':
        df_norms = df_norms[df_norms['Participant'] != 'A1CPKE6NZE67E4']
    
    # print the number of unique inputs
    unique_input = df_norms['Input.trial_'].unique()
    print(f'Number of unique inputs: {len(unique_input)}')
    print(f'Unique number of participants: {len(df_norms["Participant"].unique())}')
    
    
    ##### Split half of the participants into one dataframe and the other half into another dataframe ######
    
    # Split the df_norms into two dfs (do this 1000 times)
    lst_corrs = []
    
    # Instantiate lists to store two dataframes of size [n_items x n_it] with the mean ratings for each item in each column
    lst_df_norms1_mean = []
    lst_df_norms2_mean = []
    
    for i in tqdm(range(args.n_it)):
        df_norms1, df_norms2 = split_half(df=df_norms,
                                                random_state=i,
                                                item_col='Input.trial_',
                                                participant_col='Participant')
    
        # Compute the average of col = 'Answer.Rating_' for each Input.trial_
        df_norms1_mean = df_norms1.groupby('Input.trial_')['Answer.Rating_'].mean()
        df_norms2_mean = df_norms2.groupby('Input.trial_')['Answer.Rating_'].mean()
    
        # Append to the list
        lst_df_norms1_mean.append(df_norms1_mean)
        lst_df_norms2_mean.append(df_norms2_mean)
    
        # Get Spearman correlation between df_norms1_mean and df_norms2_mean
        corr, p = spearmanr(df_norms1_mean, df_norms2_mean)
    
        lst_corrs.append(corr)
    
    # Generate the dataframes with the mean ratings for each item in each column
    df_norms1_mean = pd.concat(lst_df_norms1_mean, axis=1)
    df_norms2_mean = pd.concat(lst_df_norms2_mean, axis=1)
    
    if args.save:
        df_norms1_mean.to_csv(join(args.SUBJECTSPLITDIR, f'exp{args.expt}_{args.norm}_norms1_mean.csv'))
        df_norms2_mean.to_csv(join(args.SUBJECTSPLITDIR, f'exp{args.expt}_{args.norm}_norms2_mean.csv'))
    
    # Get the 50 and 95% confidence intervals for the correlations
    ci = np.percentile(lst_corrs, [2.5, 50, 97.5])
    
    # Pack the results into a dataframe
    df_results = pd.DataFrame({'correlation': lst_corrs,
                               'expt': args.expt,
                                'ci_2.5': ci[0],
                                'ci_50': ci[1],
                                'ci_97.5': ci[2],
                                'n_it': args.n_it,
                                'n_items': len(df_norms1_mean),
                                'norm': args.norm,
                               'DATADIR': args.DATADIR,
                                 'EXPTFILENAME': args.EXPTFILENAME,
                                'NORMDATADIR': args.NORMDATADIR,
                                'RELIABILITYRESULTDIR': args.RELIABILITYRESULTDIR,
                                'SUBJECTSPLITDIR': args.SUBJECTSPLITDIR,
                                'save': args.save})

    if args.save:
        df_results.to_csv(join(args.RELIABILITYRESULTDIR, f'exp{args.expt}_{args.norm}_norms_reliability.csv'))
    
    # Print the results
    print(f'Correlation between df_norms1_mean and df_norms2_mean (median): {ci[1]:.3f} with 95% CI: {ci[0]:.3f} and {ci[2]:.3f}')



if __name__ == '__main__':
    main()