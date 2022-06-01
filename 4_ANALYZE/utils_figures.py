import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import os
from os.path import join
from scipy.stats import zscore
import typing

plt.rcParams['svg.fonttype'] = 'none'
now = datetime.now()
date_tag = now.strftime("%Y%m%d")

### DICT RESOURCES ###

rename_dict_expt1 = {'# meanings (human)': 'num_meanings_human',
					 '# synonyms (human)': 'num_synonyms_human',
					 '# meanings (Wordnet)': 'num_meanings_wordnet',
					 '# synonyms (Wordnet)': 'num_synonyms_wordnet',
					 'Log Subtlex frequency': 'log_subtlex_frequency',
					 'Log Subtlex CD': 'log_subtlex_cd',
					 'Arousal': 'arousal',
					 'Concreteness': 'concreteness',
					 'Familiarity': 'familiarity',
					 'Imageability': 'imageability',
					 'Valence': 'valence',
					 'GloVe distinctiveness': 'glove_distinctiveness',
					 }

rename_dict_expt2 = {'# meanings (human)': 'num_meanings_human',
					 '# synonyms (human)': 'num_synonyms_human',
					 'Google n-gram frequency': 'google_ngram_freq',
					 'Arousal': 'arousal',
					 'Concreteness': 'concreteness',
					 'Familiarity': 'familiarity',
					 'Imageability': 'imageability',
					 'Valence': 'valence', }

rename_dict_expt1_inv = {v: k for k, v in rename_dict_expt1.items()}
rename_dict_expt2_inv = {v: k for k, v in rename_dict_expt2.items()}

d_predictors = {'expt1': ['# meanings (human)', '# synonyms (human)', '# meanings (Wordnet)', '# synonyms (Wordnet)',
					  'Log Subtlex frequency', 'Log Subtlex CD',
					  'Arousal', 'Concreteness', 'Familiarity', 'Imageability', 'Valence', 'GloVe distinctiveness', ],
				'expt2': ['# meanings (human)', '# synonyms (human)', 'Google n-gram frequency',
					  'Arousal', 'Concreteness', 'Familiarity', 'Imageability', 'Valence']}

d_model_names = {'synonyms_human': '# synonyms',
				 'meanings_human': '# meanings',
				 'baseline_human': '# synonyms and # meanings',
				 'synonyms_wordnet': '# synonyms (Wordnet)',
				 'meanings_wordnet': '# meanings (Wordnet)',
				 'baseline_corpus': '# synonyms and # meanings (Wordnet)',
				 'baseline_human_arousal': 'Arousal',
				 'baseline_human_concreteness': 'Concreteness',
				'baseline_human_familiarity': 'Familiarity',
				 'baseline_human_imageability': 'Imageability',
				'baseline_human_valence': 'Valence',
				'baseline_human_log_subtlex_frequency': 'Log Subtlex frequency',
				'baseline_human_log_subtlex_cd': 'Log Subtlex CD',
				'baseline_human_glove_distinctiveness': 'GloVe distinctiveness',
				'baseline_human_google_ngram_freq': 'Google n-gram frequency',
}

d_model_colors = {'synonyms_human': '#c9e876',
				  'meanings_human': '#63ca62',
				  'baseline_human': '#46a35a',
				  'synonyms_wordnet': '#ccd1ef',
				  'meanings_wordnet': '#8d99e7',
				  'baseline_corpus': '#5563bf',}

d_acc_metrics_names = {'acc': 'Accuracy',
					   'hit.rate': 'Hit rate',
					   'false.alarm.rate': 'False alarm rate',}


### FUNCTIONS ###

def accuracy_barplot(result_dir_expt1: str,
					 result_dir_expt2: str,
					 data_subfolder: str,
					 plot_date_tag: str,
					 acc_metric: str = 'acc',
					 value_to_plot: str = 'median_CI50',
					 lower_CI_value: str = 'lower_CI2.5',
					 upper_CI_value: str = 'upper_CI97.5',
					 save: typing.Union[bool, str] = False, ):
	"""
	Plot accuracy for Expt 1 and Expt 2 with 95% CI

	Args:
		result_dir_expt1 (str): Path to the results folder
		result_dir_expt2 (str): Path to the results folder
		data_subfolder (str): Name of the data subfolder
		plot_date_tag (str): which date analyses were run on
		ceiling_subfolder (typing.Union[str, None]): Name of the ceiling subfolder
		model_name (str): Name of the model
		models_of_interest (list): List of models of interest
		value_to_plot (str): Name of the value to plot
		lower_CI_value (str): Name of the lower CI value
		upper_CI_value (str): Name of the upper CI value
		save (bool): Whether to save the figure
	"""
	
	# Load accuracies
	accs_expt1 = pd.read_csv(result_dir_expt1 +
							 data_subfolder +
							 f'/expt1_acc_metrics_{plot_date_tag}.csv', index_col=0)
	accs_expt2 = pd.read_csv(result_dir_expt2 +
							 data_subfolder +
							 f'/expt2_acc_metrics_{plot_date_tag}.csv', index_col=0)
	
	accs_expts = pd.DataFrame({'expt1': accs_expt1[acc_metric], 'expt2': accs_expt2[acc_metric]})
	
	# Plot accuracies as barplots for expt1 and expt2
	
	# Matplotlib plots error bars relative to the data, so subtract lower CI from the median, and subtract median from upper CI
	lower_CI = accs_expts.loc[value_to_plot] - accs_expts.loc[lower_CI_value]
	upper_CI = accs_expts.loc[upper_CI_value] - accs_expts.loc[value_to_plot]
	
	yerr = np.array([lower_CI.values, upper_CI.values])  # Expt1 in first col, expt2 in second col
	
	fig, ax = plt.subplots(figsize=(4, 7))
	ax.set_box_aspect(1.8)
	plt.bar([0, 1], accs_expts.loc[value_to_plot], color='grey', width=0.6, yerr=yerr)
	plt.ylim([0, 1])
	plt.xticks([0, 1], ['Expt 1', 'Expt 2'], fontsize=15)
	plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=15)
	plt.ylabel('Accuracy', fontsize=15)
	plt.tight_layout()
	if save:
		plt.savefig(save + f'expt1_expt2_{acc_metric}_barplot_{plot_date_tag}.png', dpi=300)
		plt.savefig(save + f'expt1_expt2_{acc_metric}_barplot_{plot_date_tag}.svg', dpi=300)
	plt.show()
	
	
def model_performance_across_models(result_dir: str,
									data_subfolder: str,
									plot_date_tag: str,
									ceiling_subfolder: typing.Union[str, None],
									model_name: str,
									models_of_interest: list,
									value_to_plot: str = 'median_CI50_spearman',
									lower_CI_value: str = 'lower_CI2.5_spearman',
									upper_CI_value: str = 'upper_CI97.5_spearman',
									save: typing.Union[bool, str] = False, ):
	"""
	Plot CV model performance (value_to_plot) as barplots for each model of interest.

	Args:
		result_dir (str): Path to the results folder
		data_subfolder (str): Path to the folder containing model performance data
		plot_date_tag (str): Date tag for the plots
		ceiling_subfolder (str, None): Path to the ceiling folder or None if no ceiling
		model_name (str): Name of the csv file where the models where stored (i.e., NAME-{})
		models_of_interest (list): List of strings of specific models to plot
		value_to_plot (str): Which value to plot for the model performance data. Default: 'median_CI50_spearman'
							by default, lower_CI2.5 and upper_CI97.5 are used as error bars.
		lower_CI_value (str): Which value to use for the lower CI. Default: 'lower_CI2.5_spearman'
		upper_CI_value (str): Which value to use for the upper CI. Default: 'upper_CI97.5_spearman'
		save (bool, str): Whether to save the plot or not. Default: False

	"""
	experiment_name = result_dir.split('/')[1]
	
	# Load subject ceiling data
	if ceiling_subfolder:
		subject_split_expt = pd.read_csv(result_dir +
										 ceiling_subfolder +
										 f'/subject_split_CI_{plot_date_tag}.csv', index_col=0)
	
	# Load CV accuracies
	cv_expt = pd.read_csv(result_dir +
						  data_subfolder +
						  '/cv_summary_preds' +
						  f'/across-models_df_cv_NAME-{model_name}_demeanx-True_demeany-True_permute-False_{plot_date_tag}.csv',
						  index_col=0)
	
	values_to_plot = cv_expt.loc[models_of_interest, value_to_plot]
	
	lower_CI_expt = cv_expt.loc[models_of_interest, value_to_plot] - cv_expt.loc[models_of_interest, lower_CI_value]
	upper_CI_expt = cv_expt.loc[models_of_interest, upper_CI_value] - cv_expt.loc[models_of_interest, value_to_plot]
	
	yerr_expt = np.array([lower_CI_expt.values, upper_CI_expt.values]) # first row is lower CI, second row is upper CI
	# columns are models of interest
	
	##############################################################################################################
	# assert that error bars are correct by performing a 'manual' sanity check
	test_model = models_of_interest[0]
	# Get value and upper and lower CI for the test model
	test_lower_CI = lower_CI_expt.loc[test_model]
	test_upper_CI = upper_CI_expt.loc[test_model]
	test_lower_CI_yerr = yerr_expt[:, models_of_interest.index(test_model)][0] # index into model (col) and then first row for lower CI
	test_upper_CI_yerr = yerr_expt[:, models_of_interest.index(test_model)][1] # index into model (col) and then second row for upper CI
	cv_expt_check_lower = cv_expt.loc[test_model, value_to_plot] - cv_expt.loc[test_model, lower_CI_value]
	cv_expt_check_upper = cv_expt.loc[test_model, upper_CI_value] - cv_expt.loc[test_model, value_to_plot]
	assert(test_lower_CI == test_lower_CI_yerr).all()
	assert(test_upper_CI == test_upper_CI_yerr).all()
	assert(cv_expt_check_lower == test_lower_CI).all()
	assert(cv_expt_check_upper == test_upper_CI).all()
	##############################################################################################################
	
	pretty_model_names = [d_model_names[model] for model in models_of_interest]
	model_colors = [d_model_colors[model] for model in models_of_interest]
	num_models = np.arange(len(models_of_interest))
	
	fig, ax = plt.subplots(figsize=(4, 7))
	ax.set_box_aspect(1.6)
	plt.bar(num_models,
			values_to_plot,
			yerr=yerr_expt,
			color=model_colors, width=0.6, )
	plt.ylim([0, 1])
	plt.xlim([-0.5, len(models_of_interest) - 0.5])
	plt.xticks(num_models, pretty_model_names, fontsize=16, rotation=45)
	plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=15)
	plt.ylabel('Cross-validated model performance', fontsize=15)
	plt.title(f'{experiment_name}', fontsize=16)
	
	# add a horizontal line with shaded regions
	if ceiling_subfolder:
		ax.axhline(y=subject_split_expt['median_CI50_spearman'].values, color='darkgrey', zorder=2)
		ax.fill_between([num_models[0] - 0.6, num_models[-1] + 0.6], np.repeat(subject_split_expt['lower_CI2.5_spearman'].values, 2),
						np.repeat(subject_split_expt['upper_CI97.5_spearman'].values, 2),
						color='gainsboro', alpha=0.8)
	plt.tight_layout()
	if save:
		model_savestr = '-'.join(models_of_interest)
		plt.savefig(save + f'{experiment_name}_{value_to_plot}_{model_savestr}_{plot_date_tag}.png', dpi=300)
		plt.savefig(save + f'{experiment_name}_{value_to_plot}_{model_savestr}_{plot_date_tag}.svg', dpi=300)
	plt.show()
	
def additional_predictor_increase(result_dir: str,
								  data_subfolder: str,
								  plot_date_tag: str,
								  model_name: str,
								  models_of_interest: list,
								  value_to_plot: str = 'median_CI50_spearman',
								  baseline_model: str = 'baseline_human',
								  save: typing.Union[bool, str] = False, ):
	"""
	Plot percent increase from a baseline model to models of interest (models with additional predictor).
	
	Args:
		result_dir (str): Path to the results folder
		data_subfolder (str): Path to the folder containing model performance data
		plot_date_tag (str): Date tag for the plots
		model_name (str): Name of the csv file where the models where stored (i.e., NAME-{})
		models_of_interest (list): List of strings of specific models to plot
		value_to_plot (str): Which value to plot for the model performance data. Default: 'median_CI50_spearman'
		baseline_model (str): Name of the baseline model. Default: 'baseline_human'
		save (bool, str): Whether to save the plot or not. Default: False

	Returns:

	"""
	experiment_name = result_dir.split('/')[1]

	# Load model performance CV
	cv_expt = pd.read_csv(result_dir +
						  data_subfolder +
						  '/cv_summary_preds' +
						  f'/across-models_df_cv_NAME-{model_name}_demeanx-True_demeany-True_permute-False_{plot_date_tag}.csv',
						  index_col=0)
	
	# Get percent increase from baseline_human model for value_to_plot (default: median_CI50_spearman)
	
	cv_expt_increase = (cv_expt.loc[models_of_interest, value_to_plot] - cv_expt.loc[
		'baseline_human', value_to_plot]) / \
					   cv_expt.loc[baseline_model, value_to_plot] * 100
	
	pretty_model_names = [d_model_names[model] for model in cv_expt_increase.index.values]
	
	# Plot as heatmap with colorbar
	fig, ax = plt.subplots(figsize=(8, 6))
	plt.imshow(cv_expt_increase.values.reshape(1, -1),
			   cmap='hot', vmin=0, vmax=15,
			   interpolation=None,
			   )
	plt.colorbar()
	plt.xticks(np.arange(len(models_of_interest)), pretty_model_names, rotation=90, fontsize=12)
	ax.get_yaxis().set_visible(False)
	plt.tight_layout(w_pad=4)
	plt.title(f'{experiment_name}', fontsize=16)
	if save:
		if save:
			plt.savefig(save + f'{experiment_name}_{value_to_plot}_increase-from-{baseline_model}_{plot_date_tag}.png', dpi=300)
			plt.savefig(save + f'{experiment_name}_{value_to_plot}_increase-from-{baseline_model}_{plot_date_tag}.svg', dpi=300)
	plt.show()

### TO DELETE ###
	#
	# # Load subject ceiling data
	# subject_split_expt1 = pd.read_csv(RESULTDIR_expt1 +
	# 								  subfolder +
	# 								  f'/subject_split_CI_{plot_date_tag}.csv', index_col=0)
	# subject_split_expt2 = pd.read_csv(RESULTDIR_expt2 +
	# 								  subfolder +
	# 								  f'/subject_split_CI_{plot_date_tag}.csv', index_col=0)