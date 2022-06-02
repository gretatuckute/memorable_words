import copy
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
np.random.seed(0)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

RESULT_subdirs = ['data_with_preprocessed_cols_used_for_analyses',
				  '0_subject_consistency',
				  '1_acc_metrics',
				  '2_monogamous_meanings',
				  '3_additional_predictors',
				  '4_stepwise_regression',
				  'corr_heatmaps',
				  'corr_predictors',
				  'posthoc_bootstrap']

RESULT_subfolders = ['cv_all_preds', # cross-validated model performance, across all cross-validation splits
					 'cv_summary_preds', # summary of cross-validated model performance
					 'full_summary'] # summary of model fitted on all data (i.e., not cross-validated)

### DICT RESOURCES ###
d_model_name_predictors = {'meanings_human': ['num_meanings_human'],
						   'synonyms_human': ['num_synonyms_human'],
						   'baseline_human': ['num_meanings_human', 'num_synonyms_human'],
						   'meanings_wordnet': ['num_meanings_wordnet'],
						   'synonyms_wordnet': ['num_synonyms_wordnet'],
						   'baseline_corpus': ['num_meanings_wordnet', 'num_synonyms_wordnet']}

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


### FUNCTIONS ###
def create_result_directories(result_dir: str,
							  subdirs: list,
							  subfolders: list):
	"""
	Create directories for saving results.

	Args:
		result_dir (str): directory to save results
		subdirs (list): list of subdirectories to be created
		subfolders (list): list of subfolders to be created for result subdirectories that contain model outputs

	Returns:
		None

	"""
	
	print(f'Checking whether the directory {result_dir} exists & relevant subdirectories are created...\n')
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
		print(f'Directory {result_dir} created.')
	
	# Make subdirectories
	for subdir in subdirs:
		if not os.path.exists(result_dir + subdir):
			os.makedirs(result_dir + subdir, exist_ok=True)
			print(f'Directory {result_dir + subdir} created.')
	
	for subdir in subdirs:
		# If the subdirectory starts with a number (besides 0 and 1), create subdirectories for output from linear models
		if subdir[0] in ['2', '3', '4']:  # subdirectories containing model results
			for subfolder in subfolders:
				if not os.path.exists(result_dir + subdir + '/' + subfolder):
					os.makedirs(result_dir + subdir + '/' + subfolder, exist_ok=True)
					print(f'Directory {result_dir + subdir + "/" + subfolder} created.')
					
					
def load_data(fname: str,
			fname_accs1: str,
			fname_accs2: str,):
	"""
	Load data.

	The data consists of:
	d: dataframe with all data, aggregated across subjects
	acc1: dataframe with accuracy data for half of the subjects
	acc2: dataframe with accuracy data for the other half of the subjects
	
	Assertions performed to ensure match among files.

	Args:
		fname (str): path to data file containing all data (words, i.e., items as rows, accuracy metrics and predictors as columns)
		fname_accs1 (str): path to data file containing accuracy metrics for the accuracies of half of the subjects
						   (rows are subject splits, columns are words)
		fname_accs2 (str): path to data file containing accuracy metrics for the accuracies of the other half of the subjects
						   (rows are subject splits, columns are words)

	Returns:
		d (dataframe): dataframe with all data, aggregated across subjects
		acc1 (dataframe): dataframe with accuracy data for half of the subjects, reordered to match d
		acc2 (dataframe): dataframe with accuracy data for the other half of the subjects, reordered to match d

	"""
	
	d = pd.read_csv(fname).drop(columns=['Unnamed: 0'])
	
	## Load/prep CV data ##
	# Each single row is accuracy for half of the participants, and columns are words
	acc1 = pd.read_csv(fname_accs1).drop(columns=['Unnamed: 0'])
	acc2 = pd.read_csv(fname_accs2).drop(columns=['Unnamed: 0'])
	num_splits = acc1.shape[0]
	num_words = acc1.shape[1]
	print(f'Data have {num_splits} splits, {num_words} words.\n')
	
	assert (len(np.unique(d.word_lower.values)) == acc1.shape[1])
	assert ((d.word_lower.values) == acc1.columns.values).all()
	assert (acc1.columns.values == acc2.columns.values).all()
	
	return d, acc1, acc2


def rename_predictors(df: pd.DataFrame,
					  rename_dict: dict):
	"""
	Rename columns (predictors) in a dataframe.

	Args:
		df (pd.DataFrame): df with columns to be renamed
		rename_dict (dict): dictionary with old names as keys and new names as values
		
	Returns:
		df (pd.DataFrame): df with renamed columns
		all_predictors_renamed (list): list of all predictor names after renaming
		
	"""
	df = df.rename(columns=rename_dict)
	all_predictors_renamed = list(rename_dict.values())
	
	return df, all_predictors_renamed


def minmaxscale(array: np.ndarray):
	"""
	Min-max scaling of an array.
	Args:
		array (np.ndarray): array to be scaled

	Returns:
		array (np.ndarray): scaled array

	"""
	scaler = MinMaxScaler()
	scaler.fit(array.reshape(-1, 1))
	response_scaled = scaler.transform(array.reshape(-1, 1)).ravel()
	
	return response_scaled


def preprocess_columns_in_df(df: pd.DataFrame,
							 columns_to_preprocess: typing.Union[list, np.ndarray],
							 method: str = 'demean'):
	"""
	Preprocess/transform columns in dataframe. Rename the columns with the method as the suffix.
	
	Args
		df (pd.DataFrame): dataframe with columns to be preprocessed
		columns_to_preprocess (list): list of columns (predictors) to be preprocessed
		method (str): method to be used for preprocessing
		
	Returns:
		df_preprocessed (pd.DataFrame): preprocessed dataframe only containing the columns specified in columns_to_preprocess
		
	"""
	if method == 'demean':
		df_preprocessed = df.copy()
		for predictor in columns_to_preprocess:
			df_preprocessed[predictor] = df_preprocessed[predictor].sub(df_preprocessed[predictor].mean(), axis=0)
	
	elif method == 'zscore':
		df_preprocessed = df.copy()
		for predictor in columns_to_preprocess:
			df_preprocessed[predictor] = (df_preprocessed[predictor] - df_preprocessed[predictor].mean()) / \
										 df_preprocessed[predictor].std()
	else:
		raise ValueError('Invalid method')
	
	# Append suffix _{method} to demeaned columns
	df_preprocessed = df_preprocessed[columns_to_preprocess].rename(columns=lambda x: x + '_' + method)
	
	return df_preprocessed


def get_cv_score(df: pd.DataFrame,
				 acc1: pd.DataFrame,
				 acc2: pd.DataFrame,
				 model_name: str,
				 predictors: typing.Union[list, np.ndarray] = ['# meanings (human)', '# synonyms (human)'],
				 formula: typing.Union[str, None] = None,
				 demean_x: bool = True,
				 demean_y: bool = True,
				 permute: bool = False,
				 random_seed: int = 0,
				 return_CI_summary_df: bool = True,
				 save: bool = True,
				 result_dir: str = '',
				 save_subfolder: typing.Union[bool, str] = False):
	"""
    Get cross-validation score by splitting across words (items).
    
    Fit model on the first half split of participants, and validate on the other items, but the same participants (within_subject_set),
    or different participants, different items (held_out_subject_set).
    The word split (item) is also new for every participant split iteration, i.e. for each iteration, the model is trained
    on a different set of participants and a different set of words.
    
    The function also correlates the accuracies of the two participants splits (subject_split_half_pearson/spearman_r),
    which is the consistency of human participants' performance across the two splits of participants.
    
    If demean is True (suggested!), then X and y are demeaned using the training data.
    The transformation is then applied to the held-out data.
    This enables correct estimation of the model fit based on the training data (e.g. model.rsquared) because
    this function does not fit using an intercept, which means that the data has to be demeaned in order for
    correct estimation of these parameters.
    
    Args:
        df (pd.DataFrame): dataframe with columns containing y (acc) and predictors (x) as specified in predictors
        acc1 (pd.DataFrame): accuracy for half of the participants across words
        acc2 (pd.DataFrame): accuracy for the other half of the participants across words
		model_name (str): name of the model
		predictors (list): list of predictors to use (columns in df)
		formula (str, None): if string is specified, use the statsmodels formula API to fit models
		demean_x (bool): whether to demean the predictors (based on the training set, use this transformation for the test set)
		demean_y (bool): whether to demean the y (based on the training set, use this transformation for the test set)
        permute (bool): if True, permute the words on the training set (i.e. predictors should no longer match the accuracies for that word)
		random_seed (int): random seed for the permutation if permute is True
        return_CI_summary_df (bool): if True, return a dataframe with the CI summary and not the values per split (1000 iterations)
		save (bool): if True, save the results to a csv file; save BOTH the dataframe with the CI summary and the values per split (1000 iterations)
		result_dir (str): directory to save the results to
		save_subfolder (str): subfolder to save the results to (if save is True)

    Returns:
		df_save (pd.DataFrame): dataframe with the CI summary and/or the values per split (1,000 iterations)

    """
	# For subject consistency
	subject_split_half_pearson_r = []
	subject_split_half_spearman_r = []
	
	# For model CV
	within_subject_set_pearson_r = []
	within_subject_set_spearman_r = []
	
	held_out_subject_set_pearson_r = []
	held_out_subject_set_spearman_r = []
	
	# For model CV (using the trained model on half of the participants/words)
	r2_values = []  # store the r2 of the model fitted on the training data for each split
	r2adj_values = []  # store the adjusted r2 of the model fitted on the training data for each split
	aic_values = []
	bic_values = []
	
	# Run across all subject splits:
	num_subject_splits = acc1.shape[0]
	num_words = acc1.shape[1]
	num_words_train = int(
		np.ceil(num_words / 2))  # for expt 1: 2190/2=1054.5, so round up to 1055 for training set, and 1054 for test
	
	for i in tqdm(range(num_subject_splits)):  # i is which row, i.e. which subject split to use
		
		# So we want to train on one half of the subjects (from acc1 file), on half of the words, and then test on
		# the other half of the subjects (from acc2 file), on the other words
		s1 = acc1.iloc[i]
		s2 = acc2.iloc[i]
		
		###  Obtain split half corr of subject accuracies (ceiling) ###
		r_pearson_subject_split, _ = pearsonr(s1.values, s2.values)
		r_spearman_subject_split, _ = spearmanr(s1.values, s2.values)
		
		subject_split_half_pearson_r.append(r_pearson_subject_split)
		subject_split_half_spearman_r.append(r_spearman_subject_split)
		
		# Randomly pick item (word) train/test indices
		np.random.seed(0)
		train_words_idxs = np.random.choice(num_words, size=num_words_train, replace=False, )
		test_words_idxs = np.setdiff1d(np.arange(num_words), train_words_idxs)
		
		assert (len(np.unique(
			list(test_words_idxs) + list(train_words_idxs))) == num_words)  # Make sure all words are used
		assert (len(train_words_idxs) + len(test_words_idxs) == num_words)  # Make sure all words are used
		
		# Fit model on s1 (half subjects) and half of the words (using train_words_idxs)
		# Checking whether the same words are used as predictors and accs
		train_words = df.word_lower.iloc[train_words_idxs].values
		train_words_accs = s1.iloc[train_words_idxs].index.values
		assert (train_words == train_words_accs).all()
		
		test_words = df.word_lower.iloc[test_words_idxs].values
		test_words_accs = s1.iloc[test_words_idxs].index.values
		assert (test_words == test_words_accs).all()
		
		s1_train = s1[train_words_idxs]
		s2_train = s2[train_words_idxs]
		s1_test = s1[test_words_idxs]
		s2_test = s2[test_words_idxs]
		
		# Use training indices to fit model
		if permute:
			X_train = df[predictors].iloc[train_words_idxs].sample(frac=1, random_state=random_seed)
		else:
			X_train = df[predictors].iloc[train_words_idxs]
		
		X_test = df[predictors].iloc[test_words_idxs]
		
		if demean_x:
			scaler_x = StandardScaler(with_std=False).fit(X_train)  # fit demeaning transform on train data only
			X_train = scaler_x.transform(X_train)  # demeans column-wise
			X_test = scaler_x.transform(X_test)  # Use same demeaning transform as estimated from the training set
		
		y_train_s1 = s1_train.values  # the accuracies for half of subjects (s1)
		# y_train_s2 = s2_train.values  # the accuracies for half of subjects (s2)
		y_test_s1 = s1_test.values  # within-subject split (test words)
		y_test_s2 = s2_test.values  # held-out subject splits (test words)
		
		if demean_y:
			scaler_y = StandardScaler(with_std=False).fit(
				y_train_s1.reshape(-1, 1))  # fit demeaning transform on train data only
			y_train_s1 = scaler_y.transform(
				y_train_s1.reshape(-1, 1))  # scale, even though the scaler was fitted on this
			y_test_s1 = scaler_y.transform(
				y_test_s1.reshape(-1, 1))  # Use same demeaning transform as estimated from the training set
			y_test_s2 = scaler_y.transform(y_test_s2.reshape(-1, 1))
		
		# Now X_train and y_train_s1, y_test_s1 (same subject set that, [within]) and y_test_s2 (different subject set, [held out]) ready.
		
		# Decide which model API to use. If formula string supplied, use formula API. Otherwise, use statsmodels API.
		if formula:
			# If running formula API, prep dataframes
			# Put y_train_s1 and X_train into a dataframe with X_train corresponding to the predictors (add column names)
			X_train_df = pd.DataFrame(X_train, columns=predictors)
			X_train_df['y_train_s1'] = y_train_s1
			
			# Also package X_test into a dataframe
			X_test_df = pd.DataFrame(X_test, columns=predictors)
			
			# Fit model without intercept
			model = smf.ols(f'y_train_s1 ~ {formula} -1', data=X_train_df, ).fit()
		# model_ols_formula_with_intercept = smf.ols(f'y_train_s1 ~ {formula}', data=X_train_df,).fit()
		else:
			model = sm.OLS(y_train_s1, X_train).fit()
		
		if formula:
			pred = model.predict(X_test_df)
			pred_train = model.predict(X_train_df)
		else:
			pred = model.predict(X_test)
			pred_train = model.predict(X_train)
		
		# Predict on the same set of subjects [WITHIN], but different words
		r_pearson_within_subject, p_pearson_within_subject = pearsonr(pred, y_test_s1.ravel())
		r_spearman_within_subject, p_spearman_within_subject = spearmanr(pred, y_test_s1.ravel())
		within_subject_set_pearson_r.append(r_pearson_within_subject)
		within_subject_set_spearman_r.append(r_spearman_within_subject)
		
		# Predict on different set of subjects [HELD-OUT], different words
		r_pearson_held_out_subject, p_pearson_held_out_subject = pearsonr(pred, y_test_s2.ravel())
		r_spearman_held_out_subject, p_spearman_held_out_subject = spearmanr(pred, y_test_s2.ravel())
		held_out_subject_set_pearson_r.append(r_pearson_held_out_subject)
		held_out_subject_set_spearman_r.append(r_spearman_held_out_subject)
		
		# Obtain values for model fit
		r2_values.append(model.rsquared)
		r2adj_values.append(model.rsquared_adj)
		aic_values.append(model.aic)
		bic_values.append(model.bic)
	
	df_save = pd.DataFrame({'subject_split_half_pearson_r': subject_split_half_pearson_r,
							'subject_split_half_spearman_r': subject_split_half_spearman_r,
							'within_subject_set_pearson_r': within_subject_set_pearson_r,
							'within_subject_set_spearman_r': within_subject_set_spearman_r,
							'held_out_subject_set_pearson_r': held_out_subject_set_pearson_r,
							'held_out_subject_set_spearman_r': held_out_subject_set_spearman_r,
							'model-r2_values': r2_values,  # fitted on the train data only
							'model-r2adj_values': r2adj_values,
							'model-aic_values': aic_values,
							'model-bic_values': bic_values, })
	
	# Save df with all predictions
	if save:
		df_save['predictors'] = [predictors] * len(df_save)
		df_save['demean_x'] = [demean_x] * len(df_save)
		df_save['demean_y'] = [demean_y] * len(df_save)
		df_save['permute'] = [permute] * len(df_save)
		df_save['formula'] = [formula] * len(df_save)
		
		df_save.to_csv(f'{result_dir}/{save_subfolder + "/" if save_subfolder else ""}/'
					   f'cv_all_preds/'
					   f'df_cv-all_NAME-{model_name}_demeanx-{demean_x}_demeany-{demean_y}_permute-{permute}_{date_tag}.csv')
	
	if return_CI_summary_df:
		df_save = obtain_CI(df=df_save,
							model=model_name,
							values_of_interest=['held_out_subject_set_pearson_r', 'held_out_subject_set_spearman_r',
												'model-r2_values', 'model-r2adj_values', 'model-aic_values',
												'model-bic_values'])
		if save:
			df_save['predictors'] = [predictors] * len(df_save)
			df_save['demean_x'] = [demean_x] * len(df_save)
			df_save['demean_y'] = [demean_y] * len(df_save)
			df_save['permute'] = [permute] * len(df_save)
			
			df_save.to_csv(f'{result_dir}/{save_subfolder + "/" if save_subfolder else ""}/'
						   f'cv_summary_preds/'
						   f'df_cv-summary_NAME-{model_name}_demeanx-{demean_x}_demeany-{demean_y}_permute-{permute}_{date_tag}.csv')
	
	return df_save


def obtain_CI(df: pd.DataFrame,
			  model: str,
			  values_of_interest: typing.Union[list, np.ndarray] = ['held_out_subject_set_pearson_r',
																	'held_out_subject_set_spearman_r'],
			  CI: int = 95,
			  plot_hist: bool = False, ):
	"""
	Obtain confidence intervals for the values of interest.
	
	Args:
		df (pd.DataFrame): df. Rows: number of iterations (e.g., accuracy per CV split)
							   Columns: values of interest (e.g., "held_out_subject_set_pearson_r")
		model (str): model name
		values_of_interest (list, ndarray): list of values of interest to obtain CI for
		CI (int): confidence interval limit
		plot_hist (bool): whether to plot histogram of values of interest

	Returns:
		df_save (pd.DataFrame) with CI (lower, median, upper) for the values of interest

	"""
	CI_bound_lower = (100 - CI) / 2
	CI_bound_upper = 100 - CI_bound_lower
	
	CI_lst = []
	col_names = []
	for val in values_of_interest:
		lower_CI, median, upper_CI = np.percentile(df[val].values, [CI_bound_lower, 50, CI_bound_upper])
		
		if plot_hist:
			plt.hist(df[val].values, bins=20, density=False, alpha=0.5)
			plt.axvline(x=lower_CI, color='r', linestyle='dashed', linewidth=2)
			plt.axvline(x=upper_CI, color='r', linestyle='dashed', linewidth=2)
			plt.axvline(x=median, color='k', linestyle='dashed', linewidth=2)
			plt.title(f'{model} {val} CI')
			plt.xlabel(f'{val}')
			plt.show()
		
		CI_lst.append([lower_CI, median, upper_CI])
		col_names.append([f'lower_CI{CI_bound_lower}_{val.split("_")[-2]}',
						  f'median_CI50_{val.split("_")[-2]}',
						  f'upper_CI{CI_bound_upper}_{val.split("_")[-2]}'])
	
	CI_lst_flat = [item for sublist in CI_lst for item in sublist]
	col_names_flat = [item for sublist in col_names for item in sublist]
	
	df_save = pd.DataFrame({model: CI_lst_flat}).T
	df_save.columns = col_names_flat
	
	return df_save


def compute_acc_metrics_with_error(df: pd.DataFrame,
								   result_dir: str = '',
								   save_subfolder: str = None,
								   save_str: str = None,
								   error_type: str = 'CI',
								   CI: int = 95,
								   save: bool = False, ):
	"""
	Compute accuracy metrics with error.
	Input dataframe (df), assume that columns 'acc', 'hit.rate', 'false.alarm.rate' are present.
	If error_type is CI (suggested!), compute the 50% percentile as well as the CI (as specified by CI).
	
	Args:
		df (pd.DataFrame): dataframe with columns 'acc', 'hit.rate', 'false.alarm.rate'
		result_dir (str): directory to save results
		save_subfolder (str): subfolder to save the dataframe
		save_str (str): string to save the dataframe
		error_type (str): options: 'CI'
		CI (int): confidence interval

	"""
	
	if error_type == 'CI':
		CI_bound_lower = (100 - CI) / 2
		CI_bound_upper = 100 - CI_bound_lower
		
		# Get 5%, 50%, 95% CI for accuracy, hit.rate, false.alarm.rate
		acc_ci = np.percentile(df['acc'], [CI_bound_lower, 50, CI_bound_upper])
		hit_rate_ci = np.percentile(df['hit.rate'], [CI_bound_lower, 50, CI_bound_upper])
		fa_rate_ci = np.percentile(df['false.alarm.rate'], [CI_bound_lower, 50, CI_bound_upper])
	else:
		raise ValueError('Error type not recognized')
	
	if save:
		# Save accuracy, hit.rate, false.alarm.rate and their CI
		df_acc_ci = pd.DataFrame({'acc': acc_ci,
								  'hit.rate': hit_rate_ci,
								  'false.alarm.rate': fa_rate_ci})
		df_acc_ci.index = [f'lower_CI{CI_bound_lower}', f'median_CI50', f'upper_CI{CI_bound_upper}']
		
		df_acc_ci.to_csv(f'{result_dir}/{save_subfolder + "/" if save_subfolder else ""}/'
						 f'{save_str}_{date_tag}.csv')
		print(f'Saved accuracy metrics using {error_type} {CI} to {result_dir}'
			  f'{save_subfolder + "/" if save_subfolder else ""}/'
			  f'{save_str}_{date_tag}.csv')


def get_split_half_subject_consistency(df: pd.DataFrame,
									   acc1: pd.DataFrame,
									   acc2: pd.DataFrame,
									   save: bool = False,
									   result_dir: str = '',
									   save_subfolder: str = '',
									   CI: int = 95, ):
	"""
	Compute subject consistency across subject splits of the data (which contain the accuracy for half of the subjects per word)
	
	Args:
		df (pd.DataFrame): datafrane with accuracy per word (item) and all predictors as columns
		acc1 (pd.DataFrame): dataframe with accuracy per word for subject split 1
		acc2 (pd.DataFrame): dataframe with accuracy per word for subject split 2
		save (bool): whether to save results to csv
		result_dir (str): directory to save results
		save_subfolder (str): subfolder to save results in
		CI (int): confidence interval to compute

	"""
	
	CI_bound_lower = (100 - CI) / 2
	CI_bound_upper = 100 - CI_bound_lower
	
	## Obtain split half correlation with 95% CI (could use any df, they all have the same subject corr) ##
	
	# Compute the subject split half correlations and do not return the summary df
	df_subject_split = get_cv_score(df, acc1, acc2,
									model_name='subject_constancy',
									save_subfolder='0_subject_consistency',
									save=False,
									predictors=['num_meanings_human'], return_CI_summary_df=False)
	
	plt.hist(df_subject_split.subject_split_half_spearman_r.values)
	plt.show()
	
	# Manually extract the spearman R among acc1 and acc2 and compute the CI
	low_spearman_CI, median_spearman, high_spearman_CI = np.percentile(
		df_subject_split.subject_split_half_spearman_r.values,
		[CI_bound_lower, 50, CI_bound_upper])
	
	# Store the CI values in a separate df
	df_subject_split_CI = pd.DataFrame({f'lower_CI{CI_bound_lower}_spearman': [low_spearman_CI],
										f'median_CI50_spearman': [median_spearman],
										f'upper_CI{CI_bound_upper}_spearman': [high_spearman_CI]})
	if save:
		df_subject_split_CI.to_csv(f'{result_dir}/'
								   f'{save_subfolder}/'
								   f'subject_split_CI_{date_tag}.csv')
		print(f'Subject split half correlation CI saved to {result_dir}/'
			  f'{save_subfolder}/'
			  f'subject_split_CI_{date_tag}.csv')


def stepwise_selection(X: pd.DataFrame,
					   y: typing.Union[list, np.ndarray],
					   initial_list: list = [],
					   threshold_in: float = 0.01,
					   threshold_out: float = 0.05,
					   verbose: bool = True,
					   ):
	"""Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS
    
    Args:
        X (pd.DataFrame): dataframe with candidate features (predictors) as columns
        y (list, np.ndarray): with the target values
        initial_list (list): list of features to start with (column names of X)
        threshold_in (float): include a feature if its p-value < threshold_in
        threshold_out (float): exclude a feature if its p-value > threshold_out
        verbose (bool): whether to print the sequence of inclusions and exclusions
        
    Returns:
    	included (list): list of features in the final model
    	pvalues (list): list of pvalues for the included features
    	
    Always set threshold_in < threshold_out to avoid infinite looping.
    
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    Base function from https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn/24447#24447
    """
	
	included = list(initial_list)
	while True:
		changed = False
		# forward step
		excluded = list(set(X.columns) - set(included))
		new_pval = pd.Series(index=excluded, dtype=float)
		for new_column in excluded:
			model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
			# print(f'Fitting model on: {included + [new_column]}')
			new_pval[new_column] = model.pvalues[new_column]
		best_pval = new_pval.min()
		if best_pval < threshold_in:
			best_feature = new_pval.idxmin()
			included.append(best_feature)
			changed = True
			if verbose:
				print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
		
		# backward step
		model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
		# use all coefs except intercept
		pvalues = model.pvalues.iloc[1:]
		worst_pval = pvalues.max()  # null if pvalues is empty
		if worst_pval > threshold_out:
			changed = True
			worst_feature = pvalues.idxmax()
			included.remove(worst_feature)
			if verbose:
				print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
		if not changed:
			break
	
	return included, pvalues


def get_cv_score_w_stepwise_regression(df: pd.DataFrame,
									   acc1: pd.DataFrame,
									   acc2: pd.DataFrame,
									   model_name: str,
									   predictors: typing.Union[list, np.ndarray] = ['# meanings (human)',
																					 '# synonyms (human)'],
									   demean_x: bool = True,
									   demean_y: bool = True,
									   permute: bool = False,
									   random_seed: int = 0,
									   return_CI_summary_df: bool = True,
									   save: bool = True,
									   result_dir: str = '',
									   save_subfolder: typing.Union[str, None] = ''):
	"""
    Get cross-validation score by splitting across words (items).
    
    Fit model on the first half split of participants, and validate on the other items, but the same participants (within_subject_set),
    or different participants, different items (held_out_subject_set).
    The word split (item) is also new for every participant split iteration, i.e. for each iteration, the model is trained
    on a different set of participants and a different set of words.
   
    On the training set, a forward stepwise regression is performed (to find best predictors), and then that model is
    tested on the test set (don't provide the demeaned variables, because we want to allow the transformation to be
    unbiased based on the training set).

    Args:
        df (pd.DataFrame): dataframe with columns containing y (acc) and predictors (x) as specified in predictors
        acc1 (pd.DataFrame): accuracy for half of the subjects across words
        acc2 (pd.DataFrame): accuracy for the other half of the subjects across words
		model_name (str): name of the model
		predictors (list): list of predictors to use (columns in df)
		demean_x (bool): whether to demean the predictors (based on the training set, use this transformation for the test set)
		demean_y (bool): whether to demean the y (based on the training set, use this transformation for the test set)
        permute (bool): if True, permute the words (i.e. predictors should no longer match the accuracies for that word)
		random_seed (int): random seed for the permutation if permute is True
        return_CI_summary_df (bool): if True, return a dataframe with the CI summary and not the values per split (1000 iterations)
		save (bool): if True, save the results to a csv file
		result_dir (str): directory to save the results to
		save_subfolder (str): subfolder to save the results to (if save is True)

    Returns:
		df_save (pd.DataFrame): dataframe with the CI summary and/or the values per split (1000 iterations)
		included_features_across_splits (list): list of included features (predictors, as estimed by forward regression
		 										across splits)
		 										
    """
	
	# For model CV
	within_subject_set_pearson_r = []
	within_subject_set_spearman_r = []
	
	held_out_subject_set_pearson_r = []
	held_out_subject_set_spearman_r = []
	
	# For model CV (model fitted on the training set)
	r2_values = []  # store the r2 of the model fitted on the training data for each split
	r2adj_values = []  # store the adjusted r2 of the model fitted on the training data for each split
	aic_values = []
	bic_values = []
	
	included_features_across_splits = []
	
	# Run across all subject splits:
	num_subject_splits = acc1.shape[0]
	num_words = acc1.shape[1]
	num_words_train = int(
		np.ceil(num_words / 2))  # for expt 1: 2190/2=1054.5, so round up to 1055 for training set, and 1054 for test
	
	for i in tqdm(range(num_subject_splits)):  # i is which row, i.e. which subject split to use
		
		# So we want to train on one half of the subjects (from acc1 file), on half of the words, and then test on
		# the other half of the subjects (from acc2 file), on the other words
		s1 = acc1.iloc[i]
		s2 = acc2.iloc[i]
		
		# Randomly pick train/test indices
		np.random.seed(0)
		train_words_idxs = np.random.choice(num_words, size=num_words_train, replace=False)
		test_words_idxs = np.setdiff1d(np.arange(num_words), train_words_idxs)
		
		assert (len(np.unique(
			list(test_words_idxs) + list(train_words_idxs))) == num_words)  # Make sure all words are used
		assert (len(train_words_idxs) + len(test_words_idxs) == num_words)  # Make sure all words are used
		
		# Fit model on s1 (half subjects) and half of the words (using train_words_idxs)
		# Checking whether the same words are used as predictors and accs
		train_words = df.word_lower.iloc[train_words_idxs].values
		train_words_accs = s1.iloc[train_words_idxs].index.values
		assert (train_words == train_words_accs).all()
		
		test_words = df.word_lower.iloc[test_words_idxs].values
		test_words_accs = s1.iloc[test_words_idxs].index.values
		assert (test_words == test_words_accs).all()
		
		s1_train = s1[train_words_idxs]
		s2_train = s2[train_words_idxs]
		s1_test = s1[test_words_idxs]
		s2_test = s2[test_words_idxs]
		
		# Use training indices to fit model
		if permute:
			X_train = df[predictors].iloc[train_words_idxs].sample(frac=1, random_state=random_seed)
		else:
			X_train = df[predictors].iloc[train_words_idxs]
		
		X_test = df[predictors].iloc[test_words_idxs]
		
		if demean_x:
			scaler_x = StandardScaler(with_std=False).fit(X_train)  # fit demeaning transform on train data only
			X_train = scaler_x.transform(X_train)  # demeans column-wise
			X_test = scaler_x.transform(X_test)  # Use same demeaning transform as estimated from the training set
		
		y_train_s1 = s1_train.values  # the accuracies for half of subjects (s1)
		# y_train_s2 = s2_train.values  # the accuracies for half of subjects (s2)
		y_test_s1 = s1_test.values
		y_test_s2 = s2_test.values
		
		if demean_y:
			scaler_y = StandardScaler(with_std=False).fit(
				y_train_s1.reshape(-1, 1))  # fit demeaning transform on train data only
			y_test_s1 = scaler_y.transform(
				y_test_s1.reshape(-1, 1))  # Use same demeaning transform as estimated from the training set
			y_test_s2 = scaler_y.transform(y_test_s2.reshape(-1, 1))
			y_train_s1 = scaler_y.transform(y_train_s1.reshape(-1, 1))  # also transform the training set
		
		# Initialise stepwise regression
		X_train_with_col_names = pd.DataFrame(X_train, columns=predictors)
		X_test_with_col_names = pd.DataFrame(X_test, columns=predictors)
		
		included_features, _ = stepwise_selection(X=X_train_with_col_names,
												  y=y_train_s1,
												  verbose=False)
		included_features_across_splits.append(included_features)
		
		# Now use the 'best' model (as found by stepwise regression) to obtain an unbiased estimate of the test predictivity:
		model = sm.OLS(y_train_s1, X_train_with_col_names[included_features]).fit()
		pred = model.predict(X_test_with_col_names[included_features])
		pred_train = model.predict(X_train_with_col_names[included_features])
		
		# plt.scatter(pred_train, y_train_s1)
		# plt.show()
		
		# obtain values for model fit
		r2_values.append(model.rsquared)
		r2adj_values.append(model.rsquared_adj)
		aic_values.append(model.aic)
		bic_values.append(model.bic)
		
		# Predict on the same set of subjects, but different words
		r_pearson_held_out_subject, p_pearson_within_subject = pearsonr(pred, y_test_s1.ravel())
		r_spearman_held_out_subject, p_spearman_within_subject = spearmanr(pred, y_test_s1.ravel())
		within_subject_set_pearson_r.append(r_pearson_held_out_subject)
		within_subject_set_spearman_r.append(r_spearman_held_out_subject)
		
		# Predict on different set of subjects, different words
		r_pearson_held_out_subject, p_pearson_held_out_subject = pearsonr(pred, y_test_s2.ravel())
		r_spearman_held_out_subject, p_spearman_held_out_subject = spearmanr(pred, y_test_s2.ravel())
		held_out_subject_set_pearson_r.append(r_pearson_held_out_subject)
		held_out_subject_set_spearman_r.append(r_spearman_held_out_subject)
	
	df_save = pd.DataFrame({'within_subject_set_pearson_r': within_subject_set_pearson_r,
							'within_subject_set_spearman_r': within_subject_set_spearman_r,
							'held_out_subject_set_pearson_r': held_out_subject_set_pearson_r,
							'held_out_subject_set_spearman_r': held_out_subject_set_spearman_r,
							'model-r2_values': r2_values,  # fitted on the train data only
							'model-r2adj_values': r2adj_values,
							'model-aic_values': aic_values,
							'model-bic_values': bic_values})
	
	if save:  # save df with all predictions
		df_save['predictors'] = [predictors] * len(df_save)
		df_save['demean_x'] = [demean_x] * len(df_save)
		df_save['demean_y'] = [demean_y] * len(df_save)
		df_save['permute'] = [permute] * len(df_save)
		df_save['included_features'] = included_features_across_splits
		
		df_save.to_csv(f'{result_dir}/{save_subfolder + "/" if save_subfolder else ""}/'
					   f'cv_all_preds/'
					   f'df_cv-all_NAME-{model_name}_demeanx-{demean_x}_demeany-{demean_y}_permute-{permute}_{date_tag}.csv')
	
	if return_CI_summary_df:  # save df with summary stats
		df_save = obtain_CI(df_save, model_name,
							values_of_interest=['held_out_subject_set_pearson_r', 'held_out_subject_set_spearman_r',
												'model-r2_values', 'model-r2adj_values', 'model-aic_values',
												'model-bic_values'])
		
		if save:
			df_save['predictors'] = [predictors] * len(df_save)
			df_save['demean_x'] = [demean_x] * len(df_save)
			df_save['demean_y'] = [demean_y] * len(df_save)
			df_save['permute'] = [permute] * len(df_save)
			
			# Analyze the most frequently occurring model
			u, c = np.unique(included_features_across_splits, return_counts=True)
			argmax_c = np.argmax(c)
			freq_feature_list = u[argmax_c]
			df_save['most_occurring_included_feature_list'] = [freq_feature_list] * len(df_save)
			
			df_save.to_csv(f'{result_dir}/{save_subfolder + "/" if save_subfolder else ""}/'
						   f'cv_summary_preds/'
						   f'df_cv-summary_NAME-{model_name}_demeanx-{demean_x}_demeany-{demean_y}_permute-{permute}_{date_tag}.csv')
			print(f'Saved stepwise df to: {result_dir}/{save_subfolder + "/" if save_subfolder else ""}'
				  f'cv_summary_preds/'
				  f'df_cv-summary_NAME-{model_name}_demeanx-{demean_x}_demeany-{demean_y}_permute-{permute}_{date_tag}.csv')
	
	return df_save, included_features_across_splits


def most_frequent_models(included_features_across_splits: list,
						 num_models_to_report: int = 5,):
	"""
	Find the most frequently occurring models across all splits

	Args:
		included_features_across_splits (list): list of lists of included features
		num_models_to_report (int): number of models to report

	Returns:
		df_most_frequent_models (pd.DataFrame): dataframe with the most frequent models and how many times they occur
	"""
	
	u, c = np.unique(included_features_across_splits, return_counts=True)
	
	# Get the most frequent model (based on the number of times it was included), second most frequent, etc.
	most_frequent_models = u[np.argsort(c)[::-1][:num_models_to_report]]
	most_frequent_models_counts = c[np.argsort(c)[::-1][:num_models_to_report]]
	
	# package into df
	df_most_frequent_models = pd.DataFrame(
		{'model': most_frequent_models,
		 'count': most_frequent_models_counts})
	
	return df_most_frequent_models


def bootstrap_wrapper(result_dir: str,
					  save_subfolder1: str,
					  save_subfolder2: str,
					  model1: str,
					  model2: str,
					  datetag: str,
					  demeanx: bool=True,
					  demeany: bool=True,
					  n_bootstrap: int=1000,
					  val_of_interest: str='held_out_subject_set_spearman_r',
					  save: bool=True,):
	"""
	Wrapper for bootstrapping.
	
	Args:
		save_subfolder1 (str): subfolder to fetch data from for model1
		save_subfolder2 (str): subfolder to fetch data from for model2
		model1 (str): name of model1
		model2 (str): name of model2
		datetag (str): date tag to append to filename
		demeanx (bool): whether to demean x
		demeany (bool): whether to demean y
		n_bootstrap (int): number of bootstrap samples to take
		val_of_interest (str): which value to use for bootstrapping
		save (bool): whether to save statistics based on bootstrapping to csv

	"""
	
	# Loop into the save subfolder of interest, and fetch the CV prediction values for model 1 and model 2
	df_cv_all_model1 = pd.read_csv(f'{result_dir}/'
								   f'{save_subfolder1}/'
								   f'cv_all_preds/'
								   f'df_cv-all_NAME-{model1}_demeanx-{demeanx}_demeany-{demeany}_permute-False_{datetag}.csv')
	
	df_cv_all_model2 = pd.read_csv(f'{result_dir}/'
								   f'{save_subfolder2}/'
								   f'cv_all_preds/'
								   f'df_cv-all_NAME-{model2}_demeanx-{demeanx}_demeany-{demeany}_permute-False_{datetag}.csv')
	
	df_stats = pairwise_model_comparison_comp_boostrap(model1_val=df_cv_all_model1[val_of_interest].values,
													   model2_val=df_cv_all_model2[val_of_interest].values,
													   n_bootstrap=n_bootstrap)
	
	# Save boostrap stats
	if save:
		df_stats['model1'] = model1
		df_stats['model2'] = model2
		df_stats['demeanx'] = demeanx
		df_stats['demeany'] = demeany
		df_stats['val_of_interest'] = val_of_interest
		df_stats['datetag'] = datetag
		
		# Reorder cols
		cols = df_stats.columns.values
		cols_reorder = ['model1', 'model2',
						'model1_true_mean', 'model2_true_mean',
						'true_delta_model1_model2', 'p_value', 'bootstrap_delta_mean_distrib',
						'n_bootstrap', 'demeanx', 'demeany', 'val_of_interest', 'datetag']
		df_stats = df_stats[cols_reorder]
		
		df_stats.to_csv(f'{result_dir}/'
						f'posthoc_boostrap/'
						f'boostrap-{n_bootstrap}_{model1}-{model2}_{datetag}.csv')
		print(f'Saved stats file to {result_dir}/'
			  f'posthoc_boostrap/'
			  f'boostrap-{n_bootstrap}_{model1}-{model2}_{datetag}.csv')


def pairwise_model_comparison_comp_boostrap(model1_val: typing.Union[np.ndarray, list],
											model2_val: typing.Union[np.ndarray, list],
											n_bootstrap: int=1000,
											show_distrib: bool=False,):
	"""Compare the values from two distributions (e.g., model 1 and model 2).

	Generate the true delta between the means of model1 and model2 values.
	Then generate permuted deltas from the mean of the two models' scores shuffled.
	
	Args:
		model1_val (np.ndarray, list): model 1 values
		model2_val (np.ndarray, list): model 2 values
		n_bootstrap (int): number of bootstrap samples to take
		show_distrib (bool): whether to show the distribution of the bootstrap deltas
		
	Returns:
		df_stats (pd.DataFrame): dataframe with the following columns:
					model1_true_mean (float): true mean of model1_val
					model2_true_mean (float): true mean of model2_val
					true_delta_model1_model2 (float): true delta between model1_val and model2_val
					bootstrap_delta_mean_distrib (float): mean of the bootstrap distribution
					p_value (float): p-value of the true delta
					n_bootstrap (int): number of bootstrap samples that were used
	"""
	
	true_delta = np.mean(model1_val) - np.mean(model2_val)
	
	# Run bootstrap on the value of interest for model1 vs model2 by shuffling the values
	permuted_deltas = []
	for i in tqdm(range(n_bootstrap)):
		# shuffle the model1 and model2 values and assign randomly to two lists
		model1_model2_val = np.concatenate([model1_val, model2_val])
		np.random.shuffle(model1_model2_val)
		model1_val_shuffled = model1_model2_val[:len(model1_val)]
		model2_val_shuffled = model1_model2_val[len(model1_val):]
		
		# calculate the difference between the shuffled values
		permuted_deltas.append(np.mean(model1_val_shuffled) - np.mean(model2_val_shuffled))
	
	if show_distrib:
		plt.hist(permuted_deltas)
		plt.axvline(x=true_delta, color='red')
		plt.show()
	
	# Get p-value
	p_value = np.sum(np.array(permuted_deltas) > true_delta) / n_bootstrap
	
	# Package into a df
	df_stats = pd.DataFrame({'model1_true_mean': np.mean(model1_val),
							 'model2_true_mean': np.mean(model2_val),
							 'true_delta_model1_model2': true_delta,
							 'bootstrap_delta_mean_distrib': np.mean(permuted_deltas),
							 'p_value': p_value,
							 'n_bootstrap': n_bootstrap}, index=[0])
	
	return df_stats
