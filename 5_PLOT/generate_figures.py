"""Generate figures using analyses in 4_ANALYZE"""
from utils_figures import *

RESULTDIR_expt1 = '../expt1_results/'
RESULTDIR_expt2 = '../expt2_results/'
SAVEDIR = '../expt1_expt2_results/'
plot_date_tag = '20220708' # which date analyses were run on

corr_heatmap = True
corr_predictors = True
acc_metrics = True # 1 accuracy metrics
monogamous_meanings = True # 2 monogamous meanings
additional_predictors = True # 3 additional predictors

if __name__ == '__main__':
	
	## Load data that was used to run analyses (full dataframe with rows: words, columns: target and predictors) ##
	df_expt1 = pd.read_csv(f'{RESULTDIR_expt1}/' # Load same data that was used to generate plots
						   f'data_with_preprocessed_cols_used_for_analyses/'
						   f'exp1_data_with_norms_reordered_20220708_preprocessed_cols.csv')
	df_expt2 = pd.read_csv(f'{RESULTDIR_expt2}/' # Load same data that was used to generate plots
							f'data_with_preprocessed_cols_used_for_analyses/'
						   f'exp2_data_with_norms_reordered_20220708_preprocessed_cols.csv')

	
	##### Correlation heatmap of predictors #####
	if corr_heatmap:
		save_subfolder = 'corr_heatmap'
		make_save_subfolder(SAVEDIR, save_subfolder)
	
		# Plot heatmap of predictor correlations
		for exclude_wordnet_bool in [True, False]:
			plot_full_heatmap(df=df_expt1[rename_dict_expt1_inv.keys()],  # Obtain df with predictors only,
							  title='Expt 1: Pearson correlation of norms',
							  nan_predictors=[pred for pred in rename_dict_expt2_inv if pred not in rename_dict_expt1_inv], # also run with None
							  exclude_wordnet=exclude_wordnet_bool,
							  rename_dict=rename_dict_expt1_expt2_inv,
							  plot_date_tag=plot_date_tag,
							  save_str='expt1_corr_predictors_full_heatmap',
							  save=f'{SAVEDIR}{save_subfolder}/', )
			plot_full_heatmap(df=df_expt2[rename_dict_expt2_inv.keys()],  # Obtain df with predictors only,
							  title='Expt 2: Pearson correlation of norms',
							  nan_predictors=[pred for pred in rename_dict_expt1_inv if pred not in rename_dict_expt2_inv], # also run with None
							  exclude_wordnet=exclude_wordnet_bool,
							  rename_dict=rename_dict_expt1_expt2_inv,
							  plot_date_tag=plot_date_tag,
							  save_str='expt2_corr_predictors_full_heatmap',
							  save=f'{SAVEDIR}{save_subfolder}/', )
			plot_full_heatmap(df=df_expt1[rename_dict_expt1_inv.keys()],  # Obtain df with predictors only,
							  title='Expt 1: Pearson correlation of norms',
							  nan_predictors=None, # also run with None
							  exclude_wordnet=exclude_wordnet_bool,
							  rename_dict=rename_dict_expt1_expt2_inv,
							  plot_date_tag=plot_date_tag,
							  save_str='expt1_corr_predictors_full_heatmap',
							  save=f'{SAVEDIR}{save_subfolder}/', )
			plot_full_heatmap(df=df_expt2[rename_dict_expt2_inv.keys()],  # Obtain df with predictors only,
							  title='Expt 2: Pearson correlation of norms',
							  nan_predictors=None, # also run with None
							  exclude_wordnet=exclude_wordnet_bool,
							  rename_dict=rename_dict_expt1_expt2_inv,
							  plot_date_tag=plot_date_tag,
							  save_str='expt2_corr_predictors_full_heatmap',
							  save=f'{SAVEDIR}{save_subfolder}/', )

	##### Correlation of predictors with target #####
	if corr_predictors:
		save_subfolder = 'corr_predictors'
		make_save_subfolder(SAVEDIR, save_subfolder)
	
		# For each predictor, plot the correlation with the target (accuracy) across all data points
		for normalization_setting in ['', '_demean', '_zscore']:
			for predictor in rename_dict_expt1_inv.keys():
				acc_vs_predictor(df=df_expt1,
								 predictor=predictor,
								 normalization_setting=normalization_setting,
								 rename_dict_inv=rename_dict_expt1_expt2_inv,
								 save=f'{SAVEDIR}{save_subfolder}/')
	
			for predictor in rename_dict_expt2_inv.keys():
				acc_vs_predictor(df=df_expt2,
								 predictor=predictor,
								 normalization_setting=normalization_setting,
								 rename_dict_inv=rename_dict_expt1_expt2_inv,
								 save=f'{SAVEDIR}{save_subfolder}/')


	##### Q1: How memorable are words? #####
	if acc_metrics:
		# Plot accuracies as barplots for expt1 and expt2 with 95% CI
		save_subfolder = '1_acc_metrics'
		make_save_subfolder(SAVEDIR, save_subfolder)
	
		accuracy_barplot(result_dir_expt1=RESULTDIR_expt1,
						 result_dir_expt2=RESULTDIR_expt2,
						 data_subfolder='1_acc_metrics',
						 plot_date_tag=plot_date_tag,
						 acc_metric='acc',
						 save=f'{SAVEDIR}{save_subfolder}/')


	##### Q2: How does the ideal observer ('monogamous meanings') hypothesis do? #####
	if monogamous_meanings:
		save_subfolder = '2_monogamous_meanings'
		make_save_subfolder(SAVEDIR, save_subfolder)
	
		# Human synonym/meaning predictors
		model_performance_across_models(result_dir=RESULTDIR_expt1,
										data_subfolder='2_monogamous_meanings',
										plot_date_tag=plot_date_tag,
										ceiling_subfolder='0_subject_consistency',
										model_name='baseline-human-corpus',
										models_of_interest=['synonyms_human', 'meanings_human', 'baseline_human'],
										value_to_plot='median_CI50_spearman',
										lower_CI_value='lower_CI2.5_spearman',
										upper_CI_value='upper_CI97.5_spearman',
										save=f'{SAVEDIR}{save_subfolder}/'
										)
	
		model_performance_across_models(result_dir=RESULTDIR_expt2,
										data_subfolder='2_monogamous_meanings',
										plot_date_tag=plot_date_tag,
										ceiling_subfolder='0_subject_consistency',
										model_name='baseline-human-corpus',
										models_of_interest=['synonyms_human', 'meanings_human', 'baseline_human'],
										value_to_plot='median_CI50_spearman',
										lower_CI_value='lower_CI2.5_spearman',
										upper_CI_value='upper_CI97.5_spearman',
										save=f'{SAVEDIR}{save_subfolder}/'
										)
	
		# Corpus synonym/meaning predictors (for expt 1)
		model_performance_across_models(result_dir=RESULTDIR_expt1,
										data_subfolder='2_monogamous_meanings',
										plot_date_tag=plot_date_tag,
										ceiling_subfolder='0_subject_consistency',
										model_name='baseline-human-corpus',
										models_of_interest=['synonyms_wordnet', 'meanings_wordnet', 'baseline_corpus'],
										value_to_plot='median_CI50_spearman',
										lower_CI_value='lower_CI2.5_spearman',
										upper_CI_value='upper_CI97.5_spearman',
										save=f'{SAVEDIR}{save_subfolder}/'
										)
	
		model_performance_across_models(result_dir=RESULTDIR_expt1,
										data_subfolder='2_monogamous_meanings',
										plot_date_tag=plot_date_tag,
										ceiling_subfolder='0_subject_consistency',
										model_name='baseline-human-corpus',
										models_of_interest=['synonyms_human', 'meanings_human', 'baseline_human',
															'synonyms_wordnet', 'meanings_wordnet', 'baseline_corpus'],
										value_to_plot='median_CI50_spearman',
										lower_CI_value='lower_CI2.5_spearman',
										upper_CI_value='upper_CI97.5_spearman',
										save=f'{SAVEDIR}{save_subfolder}/'
										)

	##### Q3: How do additional predictors change the accuracy? #####
	if additional_predictors:
		save_subfolder = '3_additional_predictors'
		make_save_subfolder(SAVEDIR, save_subfolder)
		
		models_of_interest_expt1 = ['baseline_human', 'baseline_human_arousal', 'baseline_human_concreteness',
							  'baseline_human_familiarity', 'baseline_human_imageability',
							  'baseline_human_valence',
									'baseline_human_google_ngram_frequency',
									'baseline_human_log_subtlex_frequency',
							  'baseline_human_log_subtlex_cd', 'baseline_human_glove_distinctiveness',]
		models_of_interest_expt2 = ['baseline_human', 'baseline_human_arousal', 'baseline_human_concreteness',
							  'baseline_human_familiarity', 'baseline_human_imageability',
							  'baseline_human_valence', 'baseline_human_google_ngram_frequency']
	
		additional_predictor_increase(result_dir=RESULTDIR_expt1,
									  data_subfolder='3_additional_predictors',
									  plot_date_tag=plot_date_tag,
									  model_name='additional-predictor',
									  models_of_interest=models_of_interest_expt1,
									  nan_models=[model for model in models_of_interest_expt2 if model not in models_of_interest_expt1],
									  value_to_plot='median_CI50_spearman',
									  baseline_model='baseline_human',
									  save=f'{SAVEDIR}{save_subfolder}/')
		
		additional_predictor_increase(result_dir=RESULTDIR_expt2,
									  data_subfolder='3_additional_predictors',
									  plot_date_tag=plot_date_tag,
									  model_name='additional-predictor',
									  models_of_interest=models_of_interest_expt2,
									  nan_models=[model for model in models_of_interest_expt1 if model not in models_of_interest_expt2],
									  value_to_plot='median_CI50_spearman',
									  baseline_model='baseline_human',
									  save=f'{SAVEDIR}{save_subfolder}/')
	
	
