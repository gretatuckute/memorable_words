from utils import *

## SETTINGS ##
plot_heatmap = True
plot_predictors_corr = True

Q0 = True # subject consistency
Q1 = True # accuracy metrics
Q2 = True # all baseline models
Q3 = True # models for baseline + ONE additional predictor
Q4 = True # forward regression

posthoc_stats = False

save = True
np.random.seed(0)

fname = "../3_PREP_ANALYSES/exp1_data_with_norms_reordered_gt_20211220.csv"
fname_accs1 = "../exp1_subject_splits/exp1_accs1_20211220.csv"
fname_accs2 = "../exp1_subject_splits/exp1_accs2_20211220.csv"
fname_word_order = "../exp1_subject_splits/word_order.txt"

if __name__ == '__main__':
	
	## Load data ##
	d = pd.read_csv(fname).drop(columns=['Unnamed: 0'])
	
	all_predictors = ['# meanings (human)', '# synonyms (human)', '# meanings (Wordnet)', '# synonyms (Wordnet)',
					  'Log Subtlex frequency', 'Log Subtlex CD',
					  'Arousal', 'Concreteness', 'Familiarity', 'Imageability', 'Valence', 'GloVe distinctiveness', ]
	
	d_predictors = d[all_predictors]
	
	## Load/prep CV data ##
	# Each single row is accuracy for half of the participants, and columns are words
	acc1 = pd.read_csv(fname_accs1).drop(columns=['Unnamed: 0'])
	acc2 = pd.read_csv(fname_accs2).drop(columns=['Unnamed: 0'])
	num_splits = acc1.shape[0]
	num_words = acc1.shape[1]
	
	# Read in word order text file
	word_order = pd.read_csv('../exp1_subject_splits/word_order.csv')
	word_order_lower = word_order['word'].str.lower()
	
	# Assert that word order txt(--> csv) file matches with the accs csv files
	assert (len(np.intersect1d(acc1.columns.values, word_order_lower.values)) == acc1.shape[1])
	
	# Sort accs by words
	acc1 = acc1[d.word_lower.values]
	acc2 = acc2[d.word_lower.values]
	
	if plot_heatmap: # Plot heatmap of predictor correlations
		plot_full_heatmap(df=d_predictors,
						  title='Expt 1: Pearson correlation of norms',
						  save_subfolder='corr_heatmaps',
						  save_str='expt1_corr_predictors_full_heatmap',
						  save=save)
		
	if Q1:
		compute_acc_metrics_with_error(df=d,
									   error_type='CI', # compute 5,50,95% CI
									   CI=95,
									   save_subfolder='1_acc_metrics',
									   save_str='expt1_acc_metrics',
									   save=save)
		
	## Fit linear models ##
	# Rename predictor names for statsmodels formula API
	df_renamed, all_predictors_renamed = rename_predictors(df=d,
						   						   		   rename_dict=rename_dict_expt1)

	### PREPROCESSING ###
	# Demean each predictor and acc (for full model fitting and evaluation of model fit)
	df_demeaned = preprocess_columns_in_df(df=df_renamed,
										   columns_to_preprocess=all_predictors_renamed + ['acc'],
										   method='demean')
	
	# Obtain names of demeaned predictors (without acc)
	all_predictors_renamed_demean = [x for x in df_demeaned.columns.values if x != 'acc']
	
	# Zscore each predictor (for visualization)
	df_zscored = preprocess_columns_in_df(df=df_renamed,
										  columns_to_preprocess=all_predictors_renamed + ['acc'],
										  method='zscore')
	
	# Merge with original df (this df contains all the preprocessed predictors)
	df = pd.concat([df_renamed, df_demeaned, df_zscored], axis=1)
	if save:
		df.to_csv(f'{RESULTDIR}/'
				  f'data_with_preprocessed_cols_used_for_analyses/'
				  f'exp1_data_with_norms_reordered_gt_20211220_preprocessed_cols.csv')
	
	if plot_predictors_corr:
		# For each predictor, plot the correlation with the acc across all data points
		for normalization_setting in ['', '_demean', '_zscore']:
		# normalization_setting = '' # _demean, _zscore or ''
		
			for predictor in all_predictors_renamed:
				acc_vs_predictor(df=df,
								 predictor=predictor,
								 normalization_setting=normalization_setting,
								 save=True,
								 save_subfolder='corr_predictors',
								 rename_dict_inv=rename_dict_expt1_inv)

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
										   save_subfolder='0_subject_consistency',
										   save=save,
										   CI=95)
	
	## Q2: Baseline models ##
	if Q2:
		save_subfolder = '2_monogamous_meanings'
		# HUMAN
		df_meanings_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
										 model_name='meanings_human',
										 predictors=['num_meanings_human'], save_subfolder=save_subfolder)
		df_synonyms_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
										 model_name='synonyms_human',
										 predictors=['num_synonyms_human'], save_subfolder=save_subfolder)
		df_baseline_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
										 model_name='baseline_human',
										 predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder=save_subfolder)
		
		# CORPUS
		df_meanings_corpus = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
										 model_name='meanings_wordnet',
										 predictors=['num_meanings_wordnet'], save_subfolder=save_subfolder)
		df_synonyms_corpus = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
										 model_name='synonyms_wordnet',
										 predictors=['num_synonyms_wordnet'], save_subfolder=save_subfolder)
		df_baseline_corpus = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
										  model_name='baseline_corpus', predictors=['num_meanings_wordnet', 'num_synonyms_wordnet'], save_subfolder=save_subfolder)
		
		# Merge all CV dfs
		df_baseline_human_corpus = pd.concat([df_meanings_human, df_synonyms_human, df_baseline_human,
											  df_meanings_corpus, df_synonyms_corpus, df_baseline_corpus,])
		
		# Fit models on the full dataset for model statistics
		# HUMAN
		m_meanings_human = smf.ols('acc_demean ~ num_meanings_human_demean', data=df).fit()
		m_synonyms_human = smf.ols('acc_demean ~ num_synonyms_human_demean', data=df).fit()
		m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean', data=df).fit()
		
		# CORPUS
		m_meanings_corpus = smf.ols('acc_demean ~ num_meanings_wordnet_demean', data=df).fit()
		m_synonyms_corpus = smf.ols('acc_demean ~ num_synonyms_wordnet_demean', data=df).fit()
		m_baseline_corpus = smf.ols('acc_demean ~ num_meanings_wordnet_demean + num_synonyms_wordnet_demean', data=df).fit()
	
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
		
		# CORPUS
		# Compare Mem ~ meanings WITH Mem ~ synonyms + meanings
		comp_meanings_baseline_corpus = sm.stats.anova_lm(m_meanings_corpus, m_baseline_corpus)
		comp_meanings_baseline_corpus['model'] = 'acc_demean ~ num_meanings_wordnet_demean'
		comp_meanings_baseline_corpus['model_add_predictor'] = 'acc_demean ~ num_meanings_wordnet_demean + num_synonyms_wordnet_demean'
		
		# Compare Mem ~ synonyms WITH Mem ~ synonyms + meanings
		comp_synonyms_baseline_corpus = sm.stats.anova_lm(m_synonyms_corpus, m_baseline_corpus)
		comp_synonyms_baseline_corpus['model'] = 'acc_demean ~ num_synonyms_wordnet_demean'
		comp_synonyms_baseline_corpus['model_add_predictor'] = 'acc_demean ~ num_meanings_wordnet_demean + num_synonyms_wordnet_demean'
		
		# Package the ANOVA model comparisons into one df
		df_comp_anova = pd.concat([comp_meanings_baseline_human, comp_synonyms_baseline_human,
								   comp_meanings_baseline_corpus, comp_synonyms_baseline_corpus], axis=0)
		
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
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_meanings_corpus_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_meanings_corpus.summary().as_text())
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_synonyms_corpus_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_synonyms_corpus.summary().as_text())
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_m_baseline_corpus_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_corpus.summary().as_text())

		# ## CD ##
		df_CD = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
							 model_name='CD', predictors=['log_subtlex_cd'], save_subfolder=save_subfolder)

		# Fit models on the full dataset for model statistics
		m_cd = smf.ols('acc_demean ~ log_subtlex_cd_demean', data=df).fit()
		
		if save:
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_cd_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_cd.summary().as_text())
		
		
	## 3. Do additional factors contribute to word memorability?
	if Q3:
		# Fit models with baseline predictors and 1 additional feature
		save_subfolder = '3_additional_predictors'
		
		# Check whether the baseline model has been run
		if 'm_baseline_human' in locals():
			pass
		else:
			df_baseline_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
											 model_name='baseline_human',
											 predictors=['num_meanings_human', 'num_synonyms_human'],) # should already have been saved in folder Q2
			m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean',
									   data=df).fit()
		
		additional_predictors = ['arousal', 'concreteness', 'familiarity', 'imageability', 'valence',
								 'log_subtlex_frequency', 'log_subtlex_cd', 'glove_distinctiveness']
		
		lst_additional_predictors_cv = []
		lst_additional_predicors_comp_anova = []
		for pred in (additional_predictors):
			model_name = f'baseline_human_{pred.lower()}'
			predictors = ['num_meanings_human', 'num_synonyms_human', pred]
			df_score = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save,
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
			lst_additional_predicors_comp_anova.append(comp)
			
			if save:
				# Store the model summaries for each individual model
				with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_{pred.lower()}_'
						  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
					fh.write(m_baseline_human_add_pred.summary().as_text())

		### CV SCORES ###
		df_additional_predictors_cv = pd.concat(lst_additional_predictors_cv)
		df_additional_predictors_cv = pd.concat([df_baseline_human, df_additional_predictors_cv])
		
		### ANOVA COMPARISON WITH BASELINE ###
		df_additional_predicors_comp_anova = pd.concat(lst_additional_predicors_comp_anova)
		
		# Create 'comparison_index' column (two rows per comparison, so repeat the index twice)
		df_additional_predicors_comp_anova['comparison_index'] = (np.repeat(np.arange(len(df_additional_predicors_comp_anova) / 2), 2))
		
		# Reorganize columns: comparison_index, model, model_add_predictor, F, Pr(>F), ss_diff, df_diff, ssr
		df_additional_predicors_comp_anova = df_additional_predictors_cv[
			['comparison_index', 'model', 'model_add_predictor', 'F', 'Pr(>F)', 'ss_diff', 'df_diff', 'ssr']]
		
		if save:
			df_additional_predictors_cv.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-additional-predictor_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
			
			df_additional_predicors_comp_anova.to_csv(f'{RESULTDIR}/{save_subfolder}/'
													  f'full_summary/'
													  f'summary_comp_anova_'
													  f'NAME-additional-predictor_'
													  f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
	
	## Forward regression ##
	if Q4:
		# If doing forward regression on the full dataset, then need to use demeaned variables:
		features_full, pvalues_full = stepwise_selection(X=X_full_demean,
														 y=y_full_demean,
														 verbose=True) # uses all predictors in X
		print(f'Resulting features based on full model fit (not CV): {features_full}')
		
		# Run within train and test splits
		df_stepwise, included_features_across_splits = \
			get_cv_score_w_stepwise_regression(df=df, acc1=acc1, acc2=acc2, save=save,
											   model_name='stepwise',
											   predictors=all_predictors_renamed,
											   save_subfolder='4_stepwise_regression')
		
		# Analyze the most frequently occurring model
		u, c = np.unique(included_features_across_splits, return_counts=True)
		argmax_c = np.argmax(c)
		freq_feature_list = u[argmax_c]
	
	if posthoc_stats:
		# Test human baseline vs CD
		bootstrap_wrapper(save_subfolder1='2_monogamous_meanings',
						  save_subfolder2='2_monogamous_meanings',
						  model1='baseline_human',
						  model2='CD',
						  datetag='20220310',
						  save=save)
		
		# Test human baseline vs num synonyms and num meanings
		bootstrap_wrapper(save_subfolder1='2_monogamous_meanings',
						  save_subfolder2='2_monogamous_meanings',
						  model1='baseline_human',
						  model2='synonyms_human',
						  datetag='20220310',
						  save=save)
		
		bootstrap_wrapper(save_subfolder1='2_monogamous_meanings',
						  save_subfolder2='2_monogamous_meanings',
						  model1='baseline_human',
						  model2='meanings_human',
						  datetag='20220310',
						  save=save)
		
		# Test models with additional predictors against the human baseline
		additional_predictors = ['arousal', 'concreteness', 'familiarity', 'imageability', 'valence',
								 'log_subtlex_frequency', 'log_subtlex_cd', 'glove_distinctiveness']
		
		for add_pred in additional_predictors:
			bootstrap_wrapper(save_subfolder1='2_monogamous_meanings',
							  save_subfolder2='3_additional_predictors',
							  model1='baseline_human',
							  model2=f'baseline_human_{add_pred}',
							  datetag='20220310',
							  save=save)

			

