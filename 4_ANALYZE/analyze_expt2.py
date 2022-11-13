from utils import *

## SETTINGS ##
Q0 = True # subject consistency
Q1 = True # accuracy metrics
Q2 = True # all baseline models
Q3 = True # models for baseline + ONE additional predictor
Q4 = True # forward-backward selection
Q5 = True # orthographic/phonological neighborhood
Q6 = True # disambiguate frequency vs meaning

posthoc_stats = False

save = True
np.random.seed(0)

fname = "../3_PREP_ANALYSES/exp2_data_with_norms_reordered_20221029.csv"
fname_accs1 = "../expt2_subject_splits/exp2_accs1_20221029.csv"
fname_accs2 = "../expt2_subject_splits/exp2_accs2_20221029.csv"

if __name__ == '__main__':
	
	## Generate folders for saving results ##
	RESULTDIR = '../expt2_results/'
	if save:
		create_result_directories(result_dir=RESULTDIR,
								  subdirs=RESULT_subdirs,
								  subfolders=RESULT_subfolders)
		
	## Load data ##
	d, acc1, acc2 = load_data(fname=fname,
							fname_accs1=fname_accs1,
							fname_accs2=fname_accs2,)
		
	if Q1: # Obtain accuracy metrics (accuracy, hit rate, false alarm rate with CI error)
		compute_acc_metrics_with_error(df=d,
									   error_type='CI_of_median',
									   CI=95,
									   result_dir=RESULTDIR,
									   save_subfolder='1_acc_metrics',
									   save_str='expt2_acc_metrics',
									   save=save)
		
	## Fit linear models ##
	# Rename predictor names for statsmodels formula API
	df_renamed, all_predictors_renamed = rename_predictors(df=d,
						   						   		   rename_dict=rename_dict_expt2)

	### PREPROCESSING ###
	# Demean each predictor and acc (for full model fitting and evaluation of model fit)
	df_demeaned = preprocess_columns_in_df(df=df_renamed,
										   columns_to_preprocess=all_predictors_renamed + ['acc'],
										   method='demean')
	
	# Obtain names of demeaned predictors (without acc)
	all_predictors_renamed_demean = [x for x in df_demeaned.columns.values if not x.startswith('acc')]
	
	# Zscore each predictor (for visualization)
	df_zscored = preprocess_columns_in_df(df=df_renamed,
										  columns_to_preprocess=all_predictors_renamed + ['acc'],
										  method='zscore')
	
	# Merge with original df (this df contains all the preprocessed predictors)
	df = pd.concat([df_renamed, df_demeaned, df_zscored], axis=1)
	if save:
		df.to_csv(f'{RESULTDIR}/'
				  f'data_with_preprocessed_cols_used_for_analyses/'
				  f'{fname.split("/")[-1].split(".")[0]}_preprocessed_cols.csv')
	
	#### RUN MODELS ####

	# Create variables based on all data (no train/test CV splits)
	X_full = df[all_predictors_renamed]
	y_full = df.acc
	
	X_full_demean = df[all_predictors_renamed_demean]
	y_full_demean = df.acc_demean
	
	## Q0: Subject consistency ##
	if Q0:
		get_split_half_subject_consistency(df=df,
										   acc1=acc1,
										   acc2=acc2,
										   result_dir=RESULTDIR,
										   save_subfolder='0_subject_consistency',
										   save=save,
										   CI=95)
	
	## Q2: Baseline models ##
	if Q2:
		save_subfolder = '2_monogamous_meanings'
		# HUMAN
		df_meanings_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
										 model_name='meanings_human',
										 predictors=['num_meanings_human'], save_subfolder=save_subfolder)
		df_synonyms_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
										 model_name='synonyms_human',
										 predictors=['num_synonyms_human'], save_subfolder=save_subfolder)
		df_baseline_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
										 model_name='baseline_human',
										 predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder=save_subfolder)
		
		# CORPUS
		df_google_ngram = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
										 model_name='google_ngram',
										 predictors=['google_ngram_frequency'], save_subfolder=save_subfolder)
		
		# Merge all CV dfs
		df_baseline_human_corpus = pd.concat([df_meanings_human, df_synonyms_human, df_baseline_human,
											  df_google_ngram])
		
		# Fit models on the full dataset for model statistics
		# HUMAN
		m_meanings_human = smf.ols('acc_demean ~ num_meanings_human_demean', data=df).fit()
		m_synonyms_human = smf.ols('acc_demean ~ num_synonyms_human_demean', data=df).fit()
		m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean', data=df).fit()
		
		# CORPUS
		m_google_ngram = smf.ols('acc_demean ~ google_ngram_frequency_demean', data=df).fit()
	
		## ANOVA COMPARISON WITH AMONG BASELINE MODELS ##
		
		# HUMAN
		# Compare Mem ~ meanings WITH Mem ~ synonyms + meanings
		comp_meanings_baseline_human = sm.stats.anova_lm(m_meanings_human, m_baseline_human)
		comp_meanings_baseline_human['model'] = 'acc_demean ~ num_meanings_human_demean'
		comp_meanings_baseline_human['model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		
		# Compare Mem ~ synonyms WITH Mem ~ synonyms + meanings
		comp_synonyms_baseline_human = sm.stats.anova_lm(m_synonyms_human, m_baseline_human)
		comp_synonyms_baseline_human['model'] = 'acc_demean ~ num_synonyms_human_demean'
		comp_synonyms_baseline_human['model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		
		# Package the ANOVA model comparisons into one df
		df_comp_anova = pd.concat([comp_meanings_baseline_human, comp_synonyms_baseline_human,], axis=0)
		
		# Create 'comparison_index' column (two rows per comparison, so repeat the index twice)
		df_comp_anova['comparison_index'] = (np.repeat(np.arange(len(df_comp_anova)/2), 2))
		
		# Reorganize columns: comparison_index, model, model_add_predictor, F, Pr(>F), ss_diff, df_diff, ssr
		df_comp_anova = df_comp_anova[['comparison_index', 'model', 'model_add_predictor', 'F', 'Pr(>F)', 'ss_diff', 'df_diff', 'ssr']]

		if save:
			df_comp_anova.to_csv(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_comp_anova_'
								 f'NAME-baseline-human-corpus_'
								 f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
		
		if save:  # store the concatenated results across all baseline models
			df_baseline_human_corpus.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-baseline-human-corpus_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
			
			# Log
			# HUMAN
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_meanings_human_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_meanings_human.summary().as_text())
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_synonyms_human_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_synonyms_human.summary().as_text())
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_human.summary().as_text())

			# CORPUS
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_google_ngram_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_google_ngram.summary().as_text())


		
	## 3. Do additional factors contribute to word memorability?
	if Q3:
		# Fit models with baseline predictors and 1 additional feature
		save_subfolder = '3_additional_predictors'
		
		# Run baseline model and if previous results exist, assert that they match the current results, any date tag
		files = os.listdir(f'{RESULTDIR}/2_monogamous_meanings/cv_summary_preds/')
		file_baseline = [f for f in files if 'df_cv-summary_NAME-baseline_human_'
						  f'demeanx-True_demeany-True_permute-False' in f]
		if len(file_baseline) >= 1:
			df_baseline_human_precomputed = pd.read_csv(f'{RESULTDIR}/2_monogamous_meanings/cv_summary_preds/'
													f'{file_baseline[-1]}', index_col=0)
		
		df_baseline_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
										 model_name='baseline_human',
										 predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder='2_monogamous_meanings') # should already have been saved in folder Q2
		m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean',
								   data=df).fit()
		
		if len(file_baseline) >= 1:
			assert(np.allclose(df_baseline_human['median_CI50_spearman'].values, df_baseline_human_precomputed['median_CI50_spearman'].values))
		
		additional_predictors = ['concreteness',  'imageability', 'familiarity','valence','arousal',
								 'google_ngram_frequency']
		
		lst_additional_predictors_cv = []
		lst_additional_predictors_comp_anova = []
		for pred in (additional_predictors):
			model_name = f'baseline_human_{pred.lower()}'
			predictors = ['num_meanings_human', 'num_synonyms_human', pred]
			df_score = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
							  		model_name=model_name, predictors=predictors, save_subfolder=save_subfolder)
			lst_additional_predictors_cv.append(df_score)

			# Fit models on the full dataset for model statistics
			# Compare human baseline model with the model with additional predictors
			m_baseline_human_add_pred = smf.ols(
				f'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + {pred}_demean', data=df).fit()

			## ANOVA COMPARISON WITH BASELINE ##
			comp = sm.stats.anova_lm(m_baseline_human, m_baseline_human_add_pred)
			comp['model'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
			comp['model_add_predictor'] = f'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + {pred}_demean'
			lst_additional_predictors_comp_anova.append(comp)
			
			if save:
				# Store the model summaries for each individual model
				with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_{pred.lower()}_'
						  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
					fh.write(m_baseline_human_add_pred.summary().as_text())

		### CV SCORES ###
		df_additional_predictors_cv = pd.concat(lst_additional_predictors_cv)
		df_additional_predictors_cv = pd.concat([df_baseline_human, df_additional_predictors_cv])
		
		### ANOVA COMPARISON WITH BASELINE ###
		df_additional_predictors_comp_anova = pd.concat(lst_additional_predictors_comp_anova)
		
		# Create 'comparison_index' column (two rows per comparison, so repeat the index twice)
		df_additional_predictors_comp_anova['comparison_index'] = (np.repeat(np.arange(len(df_additional_predictors_comp_anova) / 2), 2))
		
		# Reorganize columns: comparison_index, model, model_add_predictor, F, Pr(>F), ss_diff, df_diff, ssr
		df_additional_predictors_comp_anova = df_additional_predictors_comp_anova[
			['comparison_index', 'model', 'model_add_predictor', 'F', 'Pr(>F)', 'ss_diff', 'df_diff', 'ssr']]
		
		if save:
			df_additional_predictors_cv.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-additional-predictor_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
			
			df_additional_predictors_comp_anova.to_csv(f'{RESULTDIR}/{save_subfolder}/'
													  f'full_summary/'
													  f'summary_comp_anova_'
													  f'NAME-additional-predictor_'
													  f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
	
	## Forward-backward selection ##
	if Q4:
		save_subfolder = '4_stepwise_regression'

		# # Exclude the topic, document, orthographic, and phonological predictors
		all_predictors_renamed_no_nan_predictors = [pred for pred in all_predictors_renamed if pred not in
													['log_topic_variability', 'log_document_frequency',
													 'log_orthographic_neighborhood_size', 'log_phonological_neighborhood_size']]
		all_predictors_renamed_demean_no_nan_predictors = [pred for pred in all_predictors_renamed_demean if pred not in
														['log_topic_variability_demean', 'log_document_frequency_demean',
														 'log_orthographic_neighborhood_size_demean',
														 'log_phonological_neighborhood_size_demean']]
		
		# Run within train and test splits (CV)
		df_stepwise, included_features_across_splits = \
			get_cv_score_w_stepwise_regression(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
											   model_name='stepwise',
											   predictors=all_predictors_renamed_no_nan_predictors,
											   save_subfolder=save_subfolder)
		
		# Analyze the most frequently occurring models
		df_stepwise_most_freq = most_frequent_models(included_features_across_splits=included_features_across_splits,
													 num_models_to_report=None)  # report all models
		
		# Analyze how many times a feature was included or included as first
		df_stepwise_feature_inclusion = feature_inclusion(df_stepwise_most_freq=df_stepwise_most_freq,
														  predictors_to_check=all_predictors_renamed)
		
		if save:
			df_stepwise_most_freq.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-most-freq-stepwise-models_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
			
			df_stepwise_feature_inclusion.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-feature-inclusion-stepwise-models_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
		
		# Full model: stepwise regression on the full dataset, use demeaned variables:
		features_full, pvalues_full = stepwise_selection(X=X_full_demean[all_predictors_renamed_demean_no_nan_predictors],
														 y=y_full_demean,
														 verbose=True)  # uses all predictors in X
		print(f'Resulting features based on full model fit (not CV): {features_full}')
		
		# Fit full model using these predictors (not CV)
		model_str = 'acc_demean ~ ' + ' + '.join(features_full)
		m_stepwise = smf.ols(f'{model_str}', data=df).fit()
		
		if save:
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-stepwise_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_stepwise.summary().as_text())
	

	## Orthographical / phonological neighborhood size ##
	if Q5:
		save_subfolder = '5_orth_phon_neighborhood'

		# Get a version of the df and acc files without nans
		df_no_nan_orth_phon, acc1_no_nan_orth_phon, acc2_no_nan_orth_phon, nan_info_no_nan_orth_phon = \
			drop_nans_from_df(df=df, acc1=acc1, acc2=acc2, predictors=['log_orthographic_neighborhood_size', 'log_phonological_neighborhood_size'])

		# Save this version of the dataset
		if save:
			df_no_nan_orth_phon.to_csv(f'{RESULTDIR}/'
					  f'data_with_preprocessed_cols_used_for_analyses/'
					  f'{fname.split("/")[-1].split(".")[0]}_preprocessed_cols_no_nan_orth_phon.csv')



		df_orth = get_cv_score(df=df_no_nan_orth_phon,
							   acc1=acc1_no_nan_orth_phon, acc2=acc2_no_nan_orth_phon,
							   save=save, result_dir=RESULTDIR,
							   model_name='orth_neighborhood',
							   predictors=['log_orthographic_neighborhood_size'], save_subfolder=save_subfolder)

		df_phon = get_cv_score(df=df_no_nan_orth_phon,
							   acc1=acc1_no_nan_orth_phon, acc2=acc2_no_nan_orth_phon,
							   save=save, result_dir=RESULTDIR,
							   model_name='phon_neighborhood',
							   predictors=['log_phonological_neighborhood_size'], save_subfolder=save_subfolder)


		# Drop the words (using nan_indices) that were not included in the neighborhood size calculation
		df_baseline_human_no_nans_orth_phon = get_cv_score(df=df_no_nan_orth_phon,
														   acc1=acc1_no_nan_orth_phon, acc2=acc2_no_nan_orth_phon,
														   save=save, result_dir=RESULTDIR,
														   model_name='baseline_human_no_nans_orth_phon',
														   predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder=save_subfolder)

		# Concatenate the results
		df_orth_phon_concat = pd.concat([df_orth, df_phon, df_baseline_human_no_nans_orth_phon], axis=0)

		# Fit models on the full dataset for model statistics

		# Human baseline
		m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean', data=df_no_nan_orth_phon).fit()

		# Orthographic neighborhood size
		m_orth = smf.ols('acc_demean ~ log_orthographic_neighborhood_size_demean', data=df_no_nan_orth_phon).fit()

		# Phonological neighborhood size
		m_phon = smf.ols('acc_demean ~ log_phonological_neighborhood_size_demean', data=df_no_nan_orth_phon).fit()

		# Human baseline plus orthographic neighborhood size
		m_baseline_human_orth = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + '
										'log_orthographic_neighborhood_size_demean', data=df_no_nan_orth_phon).fit()

		# Human baseline plus phonological neighborhood size
		m_baseline_human_phon = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + '
										'log_phonological_neighborhood_size_demean', data=df_no_nan_orth_phon).fit()

		# Human baseline plus orthographic and phonological neighborhood size
		m_baseline_human_orth_phon = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + '
											 'log_orthographic_neighborhood_size_demean + '
											 'log_phonological_neighborhood_size_demean', data=df_no_nan_orth_phon).fit()


		## ANOVA COMPARISON WITH BASELINE MODELS ##

		# Compare Mem ~ synonyms + meanings WITH Mem ~ synonyms + meanings + orth_neighborhood
		comp_baseline_human_orth = sm.stats.anova_lm(m_baseline_human, m_baseline_human_orth)
		comp_baseline_human_orth['model'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		comp_baseline_human_orth[
			'model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + log_orthographic_neighborhood_size_demean'

		# Compare Mem ~ synonyms + meanings WITH Mem ~ synonyms + meanings + phon_neighborhood
		comp_baseline_human_phon = sm.stats.anova_lm(m_baseline_human, m_baseline_human_phon)
		comp_baseline_human_phon['model'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		comp_baseline_human_phon[
			'model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + log_phonological_neighborhood_size_demean'

		# Compare Mem ~ synonyms + meanings WITH Mem ~ synonyms + meanings + orth_neighborhood + phon_neighborhood
		comp_baseline_human_orth_phon = sm.stats.anova_lm(m_baseline_human, m_baseline_human_orth_phon)
		comp_baseline_human_orth_phon['model'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		comp_baseline_human_orth_phon[
			'model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + log_orthographic_neighborhood_size_demean + log_phonological_neighborhood_size_demean'

		# Package the ANOVA model comparisons into one df
		df_comp_anova = pd.concat([comp_baseline_human_orth, comp_baseline_human_phon, comp_baseline_human_orth_phon])

		# Create 'comparison_index' column (two rows per comparison, so repeat the index twice)
		df_comp_anova['comparison_index'] = (np.repeat(np.arange(len(df_comp_anova) / 2), 2))

		# Reorganize columns: comparison_index, model, model_add_predictor, F, Pr(>F), ss_diff, df_diff, ssr
		df_comp_anova = df_comp_anova[
			['comparison_index', 'model', 'model_add_predictor', 'F', 'Pr(>F)', 'ss_diff', 'df_diff', 'ssr']]

		if save:
			df_comp_anova.to_csv(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_comp_anova_'
								 f'NAME-baseline-human-orth-phon_'
								 f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')

		if save:  # store the concatenated results across all baseline models

			df_orth_phon_concat.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-baseline-human-orth-phon_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')

			# Log
			# Human baseline
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_human.summary().as_text())

			# Orthographic neighborhood size
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_orth_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_orth.summary().as_text())

			# Phonological neighborhood size
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_phon_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_phon.summary().as_text())

			# Human baseline plus orthographic neighborhood size
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_orth_' 
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_human_orth.summary().as_text())

			# Human baseline plus phonological neighborhood size
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_phon_' 
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_human_phon.summary().as_text())

			# Human baseline plus orthographic and phonological neighborhood size
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_orth_phon_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_human_orth_phon.summary().as_text())


	#### Disambiguate frequency vs meaning ####
	if Q6:
		save_subfolder = '6_freq_vs_meanings'

		freq_metrics = ['google_ngram_frequency']

		for freq_metric in freq_metrics:
			## Show frequency distribution of data
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.hist(df[freq_metric], bins=20, color='black')
			ax.set_xlabel(freq_metric)
			ax.set_ylabel('Count')
			ax.set_title(f'Distribution of {freq_metric} frequency')
			plt.tight_layout()
			plt.show()

			# Divide into 3 bins based on frequency with equal number of items in each bin
			df['frequency_bin'] = pd.qcut(df.rank(method='first')[freq_metric], 3,
													  labels=['low', 'medium', 'high'])


			# Get count of number of items in each bin and the mean and standard deviation of the frequency
			df_freq_bin = df.groupby('frequency_bin').agg(
				{freq_metric: ['count', 'mean', 'std']})
			print(df_freq_bin)

			# Get count of number of items in each bin and the mean and standard deviation of the number of meanings
			df_meanings_bin = df.groupby('frequency_bin').agg(
				{'num_meanings_human': ['count', 'mean', 'std']})
			print(df_meanings_bin)

			# Get count of number of items in each bin and the mean and standard deviation of the number of synonyms
			df_synonyms_bin = df.groupby('frequency_bin').agg(
				{'num_synonyms_human': ['count', 'mean', 'std']})
			print(df_synonyms_bin)

			# Store these stats
			if save:
				df_stats = pd.concat([df_freq_bin, df_meanings_bin, df_synonyms_bin], axis=1)
				df_stats.to_csv(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_stats_freq-bins-{freq_metric}_' 
								f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')

			# Run CV regression on data from each bin
			# First, create the new dataframes for each bin (we need to make sure the accs are also dropped in the same way)

			# Create three columns that have 1 if the item is in that bin, and 0 otherwise
			df['frequency_bin_low'] = np.where(df['frequency_bin'] == 'low', 1, np.nan)
			df['frequency_bin_medium'] = np.where(df['frequency_bin'] == 'medium', 1, np.nan)
			df['frequency_bin_high'] = np.where(df['frequency_bin'] == 'high', 1, np.nan)


			# The following will generate dfs with values only for the low, medium, and high bins
			df_low, acc1_low, acc2_low, nan_info_low = \
				drop_nans_from_df(df=df, acc1=acc1, acc2=acc2,
								  predictors=['frequency_bin_low'])

			df_medium, acc1_medium, acc2_medium, nan_info_medium = \
				drop_nans_from_df(df=df, acc1=acc1, acc2=acc2,
								  predictors=['frequency_bin_medium'])

			df_high, acc1_high, acc2_high, nan_info_high = \
				drop_nans_from_df(df=df, acc1=acc1, acc2=acc2,
								  predictors=['frequency_bin_high'])

			# Run CV regression on each bin

			## Human baseline
			# Low
			df_human_baseline_low = get_cv_score(df=df_low,
								   acc1=acc1_low, acc2=acc2_low,
								   save=save, result_dir=RESULTDIR,
								   model_name='baseline_human_low_freq_bin',
								   predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder=save_subfolder)

			# Medium
			df_human_baseline_medium = get_cv_score(df=df_medium,
									   acc1=acc1_medium, acc2=acc2_medium,
									   save=save, result_dir=RESULTDIR,
									   model_name='baseline_human_medium_freq_bin',
									   predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder=save_subfolder)

			# High
			df_human_baseline_high = get_cv_score(df=df_high,
									 acc1=acc1_high, acc2=acc2_high,
									 save=save, result_dir=RESULTDIR,
									 model_name='baseline_human_high_freq_bin',
									 predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder=save_subfolder)


			# Merge all CV dfs together
			df_freq_bins_concat = pd.concat([df_human_baseline_low, df_human_baseline_medium, df_human_baseline_high])

			if save:
				df_freq_bins_concat.to_csv(
					f'{RESULTDIR}/{save_subfolder}/'
					f'cv_summary_preds/'
					f'across-models_df_cv_NAME-freq-bins-{freq_metric}_'
					f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')








	if posthoc_stats:
		# Test human baseline vs CD
		bootstrap_wrapper(result_dir=RESULTDIR,
						  save_subfolder1='2_monogamous_meanings',
						  save_subfolder2='2_monogamous_meanings',
						  model1='baseline_human',
						  model2='CD',
						  datetag='20220310',
						  save=save)
		
		# Test human baseline vs num synonyms and num meanings
		bootstrap_wrapper(result_dir=RESULTDIR,
						  save_subfolder1='2_monogamous_meanings',
						  save_subfolder2='2_monogamous_meanings',
						  model1='baseline_human',
						  model2='synonyms_human',
						  datetag='20220310',
						  save=save)
		
		bootstrap_wrapper(result_dir=RESULTDIR,
						  save_subfolder1='2_monogamous_meanings',
						  save_subfolder2='2_monogamous_meanings',
						  model1='baseline_human',
						  model2='meanings_human',
						  datetag='20220310',
						  save=save)
		
		# Test models with additional predictors against the human baseline
		additional_predictors = ['arousal', 'concreteness', 'familiarity', 'imageability', 'valence',
								 'log_subtlex_frequency', 'log_subtlex_cd', 'glove_distinctiveness']
		
		for add_pred in additional_predictors:
			bootstrap_wrapper(result_dir=RESULTDIR,
							  save_subfolder1='2_monogamous_meanings',
							  save_subfolder2='3_additional_predictors',
							  model1='baseline_human',
							  model2=f'baseline_human_{add_pred}',
							  datetag='20220310',
							  save=save)

			

