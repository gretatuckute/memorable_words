"""Generate tables using analyses in 4_ANALYZE"""
from utils_figures import *

RESULTDIR_expt1 = '../expt1_results/'
RESULTDIR_expt2 = '../expt2_results/'
SAVEDIR = '../expt1_expt2_results/'
plot_date_tag = '20221029' # which date analyses were run on

table_1 = False
SI_table_1 = False
SI_table_2 = False
SI_table_3 = False # orth
SI_table_4 = False
SI_table_5 = False
SI_table_6 = False
SI_table_7 = False
SI_table_8 = True
SI_table_9 = True
SI_table_10 = False
SI_table_11 = False
SI_table_12 = False

if __name__ == '__main__':
	
	save_subfolder = 'tables'
	make_save_subfolder(SAVEDIR, save_subfolder)
	
	# Table 1. Ideal observer model performance.
	if table_1:
		predictor_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='2_monogamous_meanings',
								 model_name='baseline-human-corpus',
								 models_of_interest=['synonyms_human', 'meanings_human', 'baseline_human'],
								 rename_dict_models=d_model_names,
								 value_to_plot='median_CI50_spearman',
								 lower_CI_value='lower_CI2.5_spearman',
								 upper_CI_value='upper_CI97.5_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')
	
	# Table SI 1. Correlation between n=48 overlapping words
	if SI_table_1:
		overlapping_words_correlation(result_dir=SAVEDIR,
									  data_subfolder='overlapping_words',
									  plot_date_tag=plot_date_tag,)

	# Table SI 2. Ideal observer model performance, expt 1, Wordnet
	if SI_table_2:
		predictor_table_one_expt(result_dir=RESULTDIR_expt1,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='2_monogamous_meanings',
								 model_name='baseline-human-corpus',
								 models_of_interest=['synonyms_wordnet', 'meanings_wordnet', 'baseline_corpus'],
								 rename_dict_models=d_model_names,
								rename_dict_expt=d_expt_names,
								 value_to_plot='median_CI50_spearman',
								 lower_CI_value='lower_CI2.5_spearman',
								 upper_CI_value='upper_CI97.5_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')

	# Table SI 3. Model performance of ortography/phonological model (limited dataset). Expt 1 and 2.
	if SI_table_3:
		predictor_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
						plot_date_tag=plot_date_tag,
						data_subfolder='5_orth_phon_neighborhood',
						model_name='baseline-human-orth-phon',
						models_of_interest=['baseline_human_no_nans_orth_phon', 'orth_neighborhood', 'phon_neighborhood'],
						rename_dict_models=d_model_names,
						value_to_plot='median_CI50_spearman',
						lower_CI_value='lower_CI2.5_spearman',
						upper_CI_value='upper_CI97.5_spearman',
						save=f'{SAVEDIR}{save_subfolder}/')
		
	# Table SI 4. ANOVA comparison of ideal observer model performance (just using one of the predictors)
	if SI_table_4:
		ANOVA_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='2_monogamous_meanings',
								 model_name='baseline-human-corpus',
								drop_models_with='wordnet',
								 rename_dict_models=d_model_names_anova,
								 save=f'{SAVEDIR}{save_subfolder}/')


	
	# Table SI 5. Model performance with additional predictors (% increase)
	if SI_table_5:
		predictor_table_increase(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='3_additional_predictors',
								 model_name='additional-predictor',
								 models_of_interest=order_additional_predictor_models,
								 baseline_model='baseline_human',
								 rename_dict_models=d_model_names,
								 value_to_plot='median_CI50_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')
	
	# Table SI 6. Model performance with additional predictors
	if SI_table_6:
		predictor_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='3_additional_predictors',
								 model_name='additional-predictor',
								 models_of_interest=order_additional_predictor_models,
								 rename_dict_models=d_model_names,
								 value_to_plot='median_CI50_spearman',
								 lower_CI_value='lower_CI2.5_spearman',
								 upper_CI_value='upper_CI97.5_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')
		
	# Table SI 7. ANOVA comparison of baseline model and model with additional predictor
	if SI_table_7:
		ANOVA_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='3_additional_predictors',
								 model_name='additional-predictor',
								drop_models_with=None,
								rename_dict_models=d_model_names_anova,
								 save=f'{SAVEDIR}{save_subfolder}/')
		

	# Table SI 8. Model performance of frequency-chunked model: Google n-gram frequency. Expt 1 and 2.
	if SI_table_8:
		predictor_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
						plot_date_tag=plot_date_tag,
						data_subfolder='6_freq_vs_meanings',
						model_name='freq-bins-google_ngram_frequency',
						models_of_interest=['baseline_human_low_freq_bin', 'baseline_human_medium_freq_bin', 'baseline_human_high_freq_bin'],
						rename_dict_models=d_model_names,
						value_to_plot='median_CI50_spearman',
						lower_CI_value='lower_CI2.5_spearman',
						upper_CI_value='upper_CI97.5_spearman',
						save=f'{SAVEDIR}{save_subfolder}/')

	# Table SI 9. Model performance of frequency-chunked model: Log Subtlex frequency. Expt 1.
	if SI_table_9:
		predictor_table_one_expt(result_dir=RESULTDIR_expt1,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='6_freq_vs_meanings',
								 model_name='freq-bins-log_subtlex_frequency',
								 models_of_interest=['baseline_human_low_freq_bin', 'baseline_human_medium_freq_bin',
													 'baseline_human_high_freq_bin'],
								 rename_dict_models=d_model_names,
								 rename_dict_expt=d_expt_names,
								 value_to_plot='median_CI50_spearman',
								 lower_CI_value='lower_CI2.5_spearman',
								 upper_CI_value='upper_CI97.5_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')

	# Table SI 10. Model performance of contextual diversity model (full dataset). Only for expt 1.
	if SI_table_10:
		predictor_table_one_expt(result_dir=RESULTDIR_expt1,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='2_monogamous_meanings',
								 model_name='CD',
								 models_of_interest=['CD'],
								 rename_dict_models=d_model_names,
								 rename_dict_expt=d_expt_names,
								 value_to_plot='median_CI50_spearman',
								 lower_CI_value='lower_CI2.5_spearman',
								 upper_CI_value='upper_CI97.5_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')

	# Table SI 11. Model performance of topic variability AND contextual diversity model (limited dataset). Only for expt 1.
	if SI_table_11:
		predictor_table_one_expt(result_dir=RESULTDIR_expt1,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='2_monogamous_meanings',
								 model_name='baseline-human-topic',
								 models_of_interest=['baseline_human_no_nans_topic', 'CD_no_nans_topic', 'topicvar'],
								 rename_dict_models=d_model_names,
								 rename_dict_expt=d_expt_names,
								 value_to_plot='median_CI50_spearman',
								 lower_CI_value='lower_CI2.5_spearman',
								 upper_CI_value='upper_CI97.5_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')


	# Table SI 12. Stepwise selection of additional predictors and how many times they are selected
	if SI_table_12:
		stepwise_selection_count_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
									   plot_date_tag=plot_date_tag,
									   model_name='feature-inclusion-stepwise-models',
									   data_subfolder='4_stepwise_regression',
									   rename_dict_features=rename_dict_expt1_expt2_inv,
									   feature_order=order_predictors,
									   save=f'{SAVEDIR}{save_subfolder}/')










	

