"""Generate figures using analyses in 4_ANALYZE"""

from utils_figures import *

RESULTDIR_expt1 = '../expt1_results/'
RESULTDIR_expt2 = '../expt2_results/'
SAVEDIR = '../expt1_expt2_results/'
plot_date_tag = '20220519' # which date analyses were run on
save = False

if __name__ == '__main__':
	
	##### Q1: How memorable are words? #####
	# Plot accuracies as barplots for expt1 and expt2 with 95% CI
	save_subfolder = '1_acc_metrics'
	if not os.path.exists(f'{SAVEDIR}{save_subfolder}'):
		os.makedirs(f'{SAVEDIR}{save_subfolder}')
		print(f'Created directory: {save_subfolder}')

	accuracy_barplot(result_dir_expt1=RESULTDIR_expt1,
					 result_dir_expt2=RESULTDIR_expt2,
					 data_subfolder='1_acc_metrics',
					 plot_date_tag=plot_date_tag,
					 acc_metric='acc',
					 save=f'{SAVEDIR}{save_subfolder}/')
	

	##### Q2: How does the 'monogamous meanings' hypothesis do? #####
	
	save_subfolder = '2_monogamous_meanings'
	if not os.path.exists(SAVEDIR + save_subfolder):
		os.makedirs(SAVEDIR + save_subfolder)
	
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
	save_subfolder = '3_additional_predictors'
	if not os.path.exists(SAVEDIR + save_subfolder):
		os.makedirs(SAVEDIR + save_subfolder)
		
	models_of_interest_expt1 = ['baseline_human', 'baseline_human_arousal', 'baseline_human_concreteness',
						  'baseline_human_familiarity', 'baseline_human_imageability',
						  'baseline_human_valence', 'baseline_human_log_subtlex_frequency',
						  'baseline_human_log_subtlex_cd', 'baseline_human_glove_distinctiveness']
	models_of_interest_expt2 = ['baseline_human', 'baseline_human_arousal', 'baseline_human_concreteness',
						  'baseline_human_familiarity', 'baseline_human_imageability',
						  'baseline_human_valence', 'baseline_human_google_ngram_freq']
	
	additional_predictor_increase(result_dir=RESULTDIR_expt1,
								  data_subfolder='3_additional_predictors',
								  plot_date_tag=plot_date_tag,
								  model_name='additional-predictor',
								  models_of_interest=models_of_interest_expt1,
								  value_to_plot='median_CI50_spearman',
								  baseline_model='baseline_human',
								  save=f'{SAVEDIR}{save_subfolder}/')
	
	additional_predictor_increase(result_dir=RESULTDIR_expt2,
								  data_subfolder='3_additional_predictors',
								  plot_date_tag=plot_date_tag,
								  model_name='additional-predictor',
								  models_of_interest=models_of_interest_expt2,
								  value_to_plot='median_CI50_spearman',
								  baseline_model='baseline_human',
								  save=f'{SAVEDIR}{save_subfolder}/')
	
	
