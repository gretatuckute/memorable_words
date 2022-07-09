"""Generate tables using analyses in 4_ANALYZE"""
from utils_figures import *

RESULTDIR_expt1 = '../expt1_results/'
RESULTDIR_expt2 = '../expt2_results/'
SAVEDIR = '../expt1_expt2_results/'
plot_date_tag = '20220708' # which date analyses were run on

table_1 = True
SI_table_1 = True
SI_table_2 = True
SI_table_3 = True
SI_table_4 = True
SI_table_5 = True
SI_table_6 = True
SI_table_7 = True

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
		
	# Table SI 3. ANOVA comparison of ideal observer model performance (just using one of the predictors)
	if SI_table_3:
		ANOVA_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='2_monogamous_meanings',
								 model_name='baseline-human-corpus',
								drop_models_with='wordnet',
								 rename_dict_models=d_model_names_anova,
								 save=f'{SAVEDIR}{save_subfolder}/')
	
	
	# Table SI 4. Model performance with additional predictors (% increase)
	if SI_table_4:
		predictor_table_increase(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='3_additional_predictors',
								 model_name='additional-predictor',
								 models_of_interest=order_additional_predictor_models,
								 baseline_model='baseline_human',
								 rename_dict_models=d_model_names,
								 value_to_plot='median_CI50_spearman',
								 save=f'{SAVEDIR}{save_subfolder}/')
	
	# Table SI 5. Model performance with additional predictors
	if SI_table_5:
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
		
	# Table SI 6. ANOVA comparison of baseline model and model with additional predictor
	if SI_table_6:
		ANOVA_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
								 plot_date_tag=plot_date_tag,
								 data_subfolder='3_additional_predictors',
								 model_name='additional-predictor',
								drop_models_with=None,
								rename_dict_models=d_model_names_anova,
								 save=f'{SAVEDIR}{save_subfolder}/')
		
	# Table SI 7. Stepwise selection of additional predictors and how many times they are selected
	if SI_table_7:
		stepwise_selection_count_table(result_dir_expt1=RESULTDIR_expt1, result_dir_expt2=RESULTDIR_expt2,
									   plot_date_tag=plot_date_tag,
									   model_name='feature-inclusion-stepwise-models',
									   data_subfolder='4_stepwise_regression',
									   rename_dict_features=rename_dict_expt1_expt2_inv,
									   feature_order=order_predictors,
									   save=f'{SAVEDIR}{save_subfolder}/')
		
		#
	

