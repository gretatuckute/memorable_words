import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import os
from os.path import join
import typing

from resources_figures import *

sns.set_style('white')
plt.rcParams['svg.fonttype'] = 'none'
now = datetime.now()
date_tag = now.strftime("%Y%m%d")

### FUNCTIONS ###

def make_save_subfolder(SAVEDIR: str,
						save_subfolder: str,):
	"""Make a subfolder in SAVE_DIR if it does not exist.
	
	Args:
		SAVE_DIR (str): root save directory
		save_subfolder (str): subfolder
	"""
	
	if not os.path.exists(f'{SAVEDIR}{save_subfolder}'):
		os.makedirs(f'{SAVEDIR}{save_subfolder}')
		print(f'Created directory: {save_subfolder}')

def acc_vs_predictor(df: pd.DataFrame,
					 predictor: str,
					 normalization_setting: str = '',
					 save: typing.Union[bool, str] = False,
					 rename_dict_inv: dict = {},
					 plot_date_tag: str = '',
					 graphic_setting: str = 'big', ):
	"""
	Produce scatter plots of accuracy versus predictor.

	Args:
		df (pd.DataFrame): dataframe with columns 'acc' and predictor
		predictor (str): predictor to plot on x-axis
		normalization_setting (str): normalization setting to plot on x-axis
		save (bool, str): if True, save the plot, if str, save to that folder
		rename_dict_inv (dict): dictionary to rename the predictor for titles
		graphic_setting (str): graphic setting, options are: 'normal', 'big'

	Returns:

	"""
	# Get experiment name
	if len(df) == 2109:
		expt = 'expt1'
	else:
		expt = 'expt2'
	
	# Make sure that all observations are visible.
	if normalization_setting == '':
		x_limit_offset = [0.3 if predictor not in ['num_meanings_wordnet', 'num_synonyms_wordnet'] else 8]
		if predictor == 'glove_distinctiveness':
			x_limit_offset = 0.02
	elif normalization_setting == '_zscore' or normalization_setting == '_demean':
		x_limit_offset = [
			0.5 if predictor not in ['num_meanings_wordnet', 'num_synonyms_wordnet', 'google_ngram_freq'] else 1.3]
	else:
		x_limit_offset = 0.1
	
	x_min = df[f'{predictor}{normalization_setting}'].min() - x_limit_offset
	x_max = df[f'{predictor}{normalization_setting}'].max() + x_limit_offset
	
	if graphic_setting == 'normal':
		
		sns.set(style="ticks", font_scale=1.8, rc={"grid.linewidth": 1,
												   'grid.alpha': 0,  # no grid
												   "ytick.major.size": 10,
												   }, )  # 'figure.figsize':(12,12)
		xticks = np.linspace(x_min, x_max, 5)
		yticks = np.linspace(0.6, 1, 5)
		
		plt.figure(figsize=(10, 10))
		plt.xlim(x_min, x_max)
		plt.ylim(0.6, 1)
		sns.scatterplot(x=f'{predictor}{normalization_setting}', y=f'acc', data=df, s=110, alpha=0.4,
						color='black', linewidth=0,
						)
		# Add r value and plot best fit line
		r = df.corr()[f'{predictor}{normalization_setting}']['acc']
		plt.text(0.92, 0.95, f'r={r:.2f}', horizontalalignment='center', verticalalignment='center',
				 transform=plt.gca().transAxes)
		# Plot linear line with think line
		sns.regplot(x=f'{predictor}{normalization_setting}', y=f'acc', data=df, color='red',
					scatter=False, x_ci='ci', ci=95, n_boot=1000,  # use sns default CI bootstrap
					line_kws={'linewidth': 6},
					truncate=True, )
		plt.ylabel('Accuracy')
		plt.xlabel(rename_dict_inv[predictor])  # get key from rename dict
		# plt.xticks(xticks, labels=[str(x[0]) for x in np.round(xticks)])
		plt.yticks(yticks)
	
	elif graphic_setting == 'big':
		sns.set(style="ticks", font_scale=2.6, rc={"grid.linewidth": 1,
												   'grid.alpha': 0,  # no grid
												   "ytick.major.size": 10,
												   }, )  # 'figure.figsize':(12,12)
		
		xticks = np.linspace(x_min, x_max, 3).ravel()
		yticks = np.linspace(0.6, 1, 3).ravel()
		
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_box_aspect(1)
		plt.xlim(x_min, x_max)
		plt.ylim(0.55, 1)
		sns.scatterplot(x=f'{predictor}{normalization_setting}', y=f'acc', data=df, s=230, alpha=0.3,
						color='black', linewidth=0,
						)
		# Add r value and plot best fit line
		r = df.corr()[f'{predictor}{normalization_setting}']['acc']
		plt.text(0.9, 1.04, f'r={r:.2f}', horizontalalignment='center', verticalalignment='center',
				 transform=plt.gca().transAxes, fontsize=32)
		# Plot linear line with think line
		sns.regplot(x=f'{predictor}{normalization_setting}', y=f'acc', data=df, color='red',
					scatter=False, x_ci='ci', ci=95, n_boot=1000,  # use sns default CI bootstrap
					line_kws={'linewidth': 8},
					truncate=True, )
		plt.ylabel(None)  # ('Accuracy')
		plt.xlabel(None)  # (rename_dict_inv[predictor])  # get key from rename dict
		plt.xticks(xticks, labels=[str(x) for x in np.round(xticks)])
		plt.yticks(yticks)
	# plt.title(f'Accuracy vs. {rename_dict_inv[predictor]}', fontsize=30)
	
	else:
		raise ValueError('graphic_setting should be either normal or big')
	
	# plt.title(f'Accuracy vs. {rename_dict_inv[predictor]}', fontsize=22)
	plt.tight_layout()
	if save:
		normalization_setting_str = ['' if normalization_setting == '' else f'{normalization_setting}'][0]
		save_str = f'{save}/{expt}_acc_vs_{predictor}{normalization_setting_str}{plot_date_tag}'
		plt.savefig(f'{save_str}.png', dpi=120)
		plt.savefig(f'{save_str}.svg', dpi=32, )
	plt.show()


def plot_full_heatmap(df: pd.DataFrame,
					  title: str = 'Correlation heatmap',
					  nan_predictors: typing.Union[list, None] = None,
					  exclude_wordnet: bool = True,
					  plot_date_tag: str = '',
					  rename_dict: typing.Dict[str, str] = {},
					  save_str: str = 'corr_heatmap',
					  save: typing.Union[bool, str] = False, ):
	"""Correlate the input df with itself, and plot the result as a heatmap

	Args:
		df (pd.DataFrame): dataframe to be correlated
		title (str): title of the plot
		nan_predictors (list, optional): list of predictors (columns) to be added in as nan values in the heatmap
		exclude_wordnet (bool, optional): whether to exclude wordnet predictors from the heatmap
		plot_date_tag (str, optional): date tag to be added to the plot title
		rename_dict (dict, optional): dict of column names to be renamed
			Keys are current columns names, values are new column names
		save_str (str): string to be added to the file name
		save (bool, str): if True, save the plot, if str, save to that folder
	"""
	if nan_predictors is not None:
		df_copy = df.copy()
		for predictor in nan_predictors:
			df_copy.loc[:, predictor] = np.nan
			save_str_nan_models = '_with-nan-preds'
		df = df_copy
	else:
		save_str_nan_models = ''
	
	if exclude_wordnet: # Exclude wordnet predictors (has wordnet in the column name)
		df = df.loc[:, ~df.columns.str.contains('wordnet')]
		reorder_predictors = [x for x in order_predictors if 'Wordnet' not in x]
		save_str_wordnet = '_without-wordnet'
	else:
		save_str_wordnet = ''
		reorder_predictors = order_predictors
	
	# Rename columns to pretty names
	df = df.rename(columns=rename_dict, inplace=False)
	
	if nan_predictors is not None: # Reorder columns using order_predictors
		df = df[reorder_predictors]
	else: # exclude the predictors that exist in reorder_predictors that are not in the df columns
		reorder_predictors = [x for x in reorder_predictors if x in df.columns]
		df = df[reorder_predictors]
		
	df_corr = df.corr()
	
	plt.figure(figsize=(15, 15))
	sns.set(font_scale=1.6)
	g = sns.heatmap(df_corr, cmap='RdBu_r', center=0, square=True,
					cbar_kws={"shrink": 0.75}, vmin=-1, vmax=1,
					linewidths=0.005, linecolor='black')
	g.set_facecolor('white')  # which values nans will have
	plt.title(title)
	plt.tight_layout(pad=2)
	plt.xticks(rotation=55)
	if save:
		save_str_full = f'{save}/{save_str}{save_str_nan_models}{save_str_wordnet}_{plot_date_tag}'
		plt.savefig(f'{save_str_full}.png', dpi=180)
		plt.savefig(f'{save_str_full}.svg', dpi=180)
		df.corr().to_csv(f'{save_str_full}.csv')
	plt.show()


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
								  nan_models: typing.Union[list, None] = None,
								  value_to_plot: str = 'median_CI50_spearman',
								  baseline_model: str = 'baseline_human',
								  cmap: str = 'plasma',
								  save: typing.Union[bool, str] = False, ):
	"""
	Plot percent increase from a baseline model to models of interest (models with additional predictor).
	
	Args:
		result_dir (str): Path to the results folder
		data_subfolder (str): Path to the folder containing model performance data
		plot_date_tag (str): Date tag for the plots
		model_name (str): Name of the csv file where the models where stored (i.e., NAME-{})
		models_of_interest (list): List of strings of specific models to plot
		nan_models (list, optional): List of models to add in as empty, nan columns. Defaults to None.
		value_to_plot (str): Which value to plot for the model performance data. Default: 'median_CI50_spearman'
		baseline_model (str): Name of the baseline model. Default: 'baseline_human'
		cmap (str): Colormap to use for the plot. Default: 'plasma'
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
	if nan_models: # Add in empty columns for nan models
		for model in nan_models:
			cv_expt_increase[model] = np.nan
			
		# Reorder models (rows)
		cv_expt_increase = cv_expt_increase.reindex(order_additional_predictor_models, axis=1)
		
		save_str_nan_models = 'with-nan-models_'
	else:
		save_str_nan_models = ''
	
	pretty_model_names = [d_model_names[model] for model in cv_expt_increase.index.values]
	
	# Plot as heatmap with colorbar
	fig, ax = plt.subplots(figsize=(8, 6))
	plt.imshow(cv_expt_increase.values.reshape(1, -1),
			   cmap=cmap, vmin=0, vmax=10,
			   interpolation=None,
			   )
	plt.colorbar()
	plt.xticks(np.arange(len(pretty_model_names)), pretty_model_names, rotation=55, fontsize=12)
	ax.get_yaxis().set_visible(False)
	plt.tight_layout(w_pad=4)
	plt.title(f'{experiment_name}', fontsize=16)
	if save:
		save_str = f'{experiment_name}_{value_to_plot}_increase-from-{baseline_model}_{save_str_nan_models}{plot_date_tag}'
		plt.savefig(save + f'{save_str}.png', dpi=300)
		plt.savefig(save + f'{save_str}.svg', dpi=300)
		pd.DataFrame(cv_expt_increase.values,
					 index=cv_expt_increase.index.values,
					 columns=['percent_increase']).to_csv(save + f'{save_str}.csv')
	plt.show()
	
def sort_and_find_word_of_interest(df: pd.DataFrame,
								   word_of_interest: str):
	"""Sort dataframe (rows: items, columns: norms and accuracy (acc) metrics.

	Args:
		df (pd.DataFrame): Dataframe with rows: items, columns: norms and accuracy (acc) metrics.
		word_of_interest (str): Word of interest to find in the dataframe.

	Returns:
		pd.DataFrame: Row in df that corresponds to word_of_interest.
	"""
	
	df_sorted = df.sort_values('acc', ascending=False).drop(columns=['Unnamed: 0']).reset_index(drop=False)
	
	# find word
	return df_sorted.loc[df_sorted['word_lower'] == 'avalance']