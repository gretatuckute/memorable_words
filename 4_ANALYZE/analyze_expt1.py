from utils import *

## SETTINGS ##
Q0 = False # subject consistency
Q1 = False # accuracy metrics
Q2 = False # all baseline models
Q3 = False # models for baseline + ONE additional predictor
Q4 = False # forward-backward selection
Q5 = True # orthographic/phonological neighborhood
Q6 = True # disambiguate frequency vs meaning

posthoc_stats = False

save = True
np.random.seed(0)

fname = "../3_PREP_ANALYSES/exp1_data_with_norms_reordered_20221010.csv"
fname_accs1 = "../expt1_subject_splits/exp1_accs1_20221010.csv"
fname_accs2 = "../expt1_subject_splits/exp1_accs2_20221010.csv"

if __name__ == '__main__':
	
	## Generate folders for saving results ##
	RESULTDIR = '../expt1_results/'
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
		#
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

		# # CORPUS
		# df_meanings_corpus = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
		# 								 model_name='meanings_wordnet',
		# 								 predictors=['num_meanings_wordnet'], save_subfolder=save_subfolder)
		# df_synonyms_corpus = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
		# 								 model_name='synonyms_wordnet',
		# 								 predictors=['num_synonyms_wordnet'], save_subfolder=save_subfolder)
		# df_baseline_corpus = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
		# 								  model_name='baseline_corpus', predictors=['num_meanings_wordnet', 'num_synonyms_wordnet'], save_subfolder=save_subfolder)
		#
		# # Merge all CV dfs
		# df_baseline_human_corpus_concat = pd.concat([df_meanings_human, df_synonyms_human, df_baseline_human,
		# 									  df_meanings_corpus, df_synonyms_corpus, df_baseline_corpus,])
		#
		# if save:  # store the concatenated results across all baseline models
		# 	df_baseline_human_corpus_concat.to_csv(
		# 		f'{RESULTDIR}/{save_subfolder}/'
		# 		f'cv_summary_preds/'
		# 		f'across-models_df_cv_NAME-baseline-human-corpus_'
		# 		f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
		#
		# # Fit models on the full dataset for model statistics
		# # HUMAN
		# m_meanings_human = smf.ols('acc_demean ~ num_meanings_human_demean', data=df).fit()
		# m_synonyms_human = smf.ols('acc_demean ~ num_synonyms_human_demean', data=df).fit()
		# m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean', data=df).fit()
		#
		# # CORPUS
		# m_meanings_corpus = smf.ols('acc_demean ~ num_meanings_wordnet_demean', data=df).fit()
		# m_synonyms_corpus = smf.ols('acc_demean ~ num_synonyms_wordnet_demean', data=df).fit()
		# m_baseline_corpus = smf.ols('acc_demean ~ num_meanings_wordnet_demean + num_synonyms_wordnet_demean', data=df).fit()
		#
		# ## ANOVA COMPARISON WITH BASELINE MODELS ##
		#
		# # HUMAN
		# # Compare Mem ~ meanings WITH Mem ~ synonyms + meanings
		# comp_meanings_baseline_human = sm.stats.anova_lm(m_meanings_human, m_baseline_human)
		# comp_meanings_baseline_human['model'] = 'acc_demean ~ num_meanings_human_demean'
		# comp_meanings_baseline_human['model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		#
		# # Compare Mem ~ synonyms WITH Mem ~ synonyms + meanings
		# comp_synonyms_baseline_human = sm.stats.anova_lm(m_synonyms_human, m_baseline_human)
		# comp_synonyms_baseline_human['model'] = 'acc_demean ~ num_synonyms_human_demean'
		# comp_synonyms_baseline_human['model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		#
		# # CORPUS
		# # Compare Mem ~ meanings WITH Mem ~ synonyms + meanings
		# comp_meanings_baseline_corpus = sm.stats.anova_lm(m_meanings_corpus, m_baseline_corpus)
		# comp_meanings_baseline_corpus['model'] = 'acc_demean ~ num_meanings_wordnet_demean'
		# comp_meanings_baseline_corpus['model_add_predictor'] = 'acc_demean ~ num_meanings_wordnet_demean + num_synonyms_wordnet_demean'
		#
		# # Compare Mem ~ synonyms WITH Mem ~ synonyms + meanings
		# comp_synonyms_baseline_corpus = sm.stats.anova_lm(m_synonyms_corpus, m_baseline_corpus)
		# comp_synonyms_baseline_corpus['model'] = 'acc_demean ~ num_synonyms_wordnet_demean'
		# comp_synonyms_baseline_corpus['model_add_predictor'] = 'acc_demean ~ num_meanings_wordnet_demean + num_synonyms_wordnet_demean'
		#
		# # Package the ANOVA model comparisons into one df
		# df_comp_anova = pd.concat([comp_meanings_baseline_human, comp_synonyms_baseline_human,
		# 						   comp_meanings_baseline_corpus, comp_synonyms_baseline_corpus], axis=0)
		#
		# # Create 'comparison_index' column (two rows per comparison, so repeat the index twice)
		# df_comp_anova['comparison_index'] = (np.repeat(np.arange(len(df_comp_anova)/2), 2))
		#
		# # Reorganize columns: comparison_index, model, model_add_predictor, F, Pr(>F), ss_diff, df_diff, ssr
		# df_comp_anova = df_comp_anova[['comparison_index', 'model', 'model_add_predictor', 'F', 'Pr(>F)', 'ss_diff', 'df_diff', 'ssr']]
		#
		# if save:
		# 	df_comp_anova.to_csv(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_comp_anova_'
		# 						 f'NAME-baseline-human-corpus_'
		# 						 f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')
		#
		#
		# 	# Log
		# 	# HUMAN
		# 	with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_meanings_human_'
		# 			  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
		# 		fh.write(m_meanings_human.summary().as_text())
		# 	with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_synonyms_human_'
		# 			  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
		# 		fh.write(m_synonyms_human.summary().as_text())
		# 	with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_'
		# 			  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
		# 		fh.write(m_baseline_human.summary().as_text())
		#
		# 	# CORPUS
		# 	with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_meanings_corpus_'
		# 			  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
		# 		fh.write(m_meanings_corpus.summary().as_text())
		# 	with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_synonyms_corpus_'
		# 			  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
		# 		fh.write(m_synonyms_corpus.summary().as_text())
		# 	with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_m_baseline_corpus_'
		# 			  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
		# 		fh.write(m_baseline_corpus.summary().as_text())
		#
		# # ## CD ##
		df_CD = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
							 model_name='CD', predictors=['log_subtlex_cd'], save_subfolder=save_subfolder)

		# Fit models on the full dataset for model statistics
		m_cd = smf.ols('acc_demean ~ log_subtlex_cd_demean', data=df).fit()

		if save:
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_cd_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_cd.summary().as_text())

		## Topic variability (and document frequency) ##

		# Run the topic variability predictors. However, this one has NaNs, so first obtain the dataset without NaNs:
		df_no_nan_topic, acc1_no_nan_topic, acc2_no_nan_topic, nan_info_no_nan_topic = \
			drop_nans_from_df(df=df, acc1=acc1, acc2=acc2,
							  predictors=['log_topic_variability', 'log_document_frequency'])

		df_topic = get_cv_score(df=df_no_nan_topic,
							   acc1=acc1_no_nan_topic, acc2=acc2_no_nan_topic,
							   save=save, result_dir=RESULTDIR,
							   model_name='topicvar',
							   predictors=['log_topic_variability'], save_subfolder=save_subfolder)

		df_docfreq = get_cv_score(df=df_no_nan_topic,
								   acc1=acc1_no_nan_topic, acc2=acc2_no_nan_topic,
								   save=save, result_dir=RESULTDIR,
								   model_name='docfreq',
								   predictors=['log_document_frequency'], save_subfolder=save_subfolder)


		# Fit models using baseline model, but with limited set of words (due to the NaNs)
		df_baseline_human_no_nans_topic = get_cv_score(df=df_no_nan_topic,
													   acc1=acc1_no_nan_topic, acc2=acc2_no_nan_topic,
													   save=save, result_dir=RESULTDIR,
													   model_name='baseline_human_no_nans_topic',
													   predictors=['num_meanings_human', 'num_synonyms_human'],
													   save_subfolder=save_subfolder)

		# Model with baseline predictors and topic variability
		df_baseline_human_topic = get_cv_score(df=df_no_nan_topic,
											   acc1=acc1_no_nan_topic, acc2=acc2_no_nan_topic,
											   save=save, result_dir=RESULTDIR,
											   model_name='baseline_human_topic',
											   predictors=['num_meanings_human', 'num_synonyms_human',
											   				'log_topic_variability'],
											   save_subfolder=save_subfolder)

		# Also run CD on the limited set
		df_CD_no_nans_topic = get_cv_score(df=df_no_nan_topic,
										   acc1=acc1_no_nan_topic, acc2=acc2_no_nan_topic,
										   save=save, result_dir=RESULTDIR,
										   model_name='CD_no_nans_topic',
										   predictors=['log_subtlex_cd'], save_subfolder=save_subfolder)

		# Merge all CV dfs
		df_baseline_human_topic_concat = pd.concat([df_meanings_human, df_synonyms_human, df_baseline_human,
													df_CD,
													df_topic, df_docfreq, df_baseline_human_no_nans_topic, df_baseline_human_topic, df_CD_no_nans_topic],)

		if save:  # store the concatenated results across all baseline models
			df_baseline_human_topic_concat.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-baseline-human-topic_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')


		# Fit models on the full dataset for model statistics

		# Human baseline
		m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean', data=df_no_nan_topic).fit()

		# Topic variability
		m_topic = smf.ols('acc_demean ~ log_topic_variability_demean', data=df_no_nan_topic).fit()

		# Document frequency
		m_docfreq = smf.ols('acc_demean ~ log_document_frequency_demean', data=df_no_nan_topic).fit()

		# Topic variability and human baseline
		m_baseline_human_topic = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + '
										 'log_topic_variability_demean', data=df_no_nan_topic).fit()


		## ANOVA COMPARISON WITH BASELINE MODELS ##

		# Compare Mem ~ synonyms + meanings WITH Mem ~ synonyms + meanings + orth_neighborhood
		comp_baseline_human_orth = sm.stats.anova_lm(m_baseline_human, m_baseline_human_topic)
		comp_baseline_human_orth['model'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean'
		comp_baseline_human_orth[
			'model_add_predictor'] = 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + log_topic_variability_demean'


		# Package the ANOVA model comparisons into one df
		df_comp_anova = comp_baseline_human_orth

		# Create 'comparison_index' column (two rows per comparison, so repeat the index twice)
		df_comp_anova['comparison_index'] = (np.repeat(np.arange(len(df_comp_anova) / 2), 2))

		# Reorganize columns: comparison_index, model, model_add_predictor, F, Pr(>F), ss_diff, df_diff, ssr
		df_comp_anova = df_comp_anova[
			['comparison_index', 'model', 'model_add_predictor', 'F', 'Pr(>F)', 'ss_diff', 'df_diff', 'ssr']]


		if save:
			df_comp_anova.to_csv(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_comp_anova_'
								 f'NAME-baseline-human-topic_'
								 f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')


			# Log
			# Human baseline
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_human.summary().as_text())

			# Topic variability
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_topic_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_topic.summary().as_text())

			# Document frequency
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_docfreq_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_docfreq.summary().as_text())

			# Topic variability and human baseline
			with open(f'{RESULTDIR}/{save_subfolder}/full_summary/summary_NAME-m_baseline_human_topic_'
					  f'demeanx-True_demeany-True_permute-False_{date_tag}.txt', 'w') as fh:
				fh.write(m_baseline_human_topic.summary().as_text())



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
													f'{file_baseline[0]}', index_col=0)
		
		df_baseline_human = get_cv_score(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
										 model_name='baseline_human',
										 predictors=['num_meanings_human', 'num_synonyms_human'], save_subfolder='2_monogamous_meanings') # should already have been saved in folder Q2
		m_baseline_human = smf.ols('acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean',
								   data=df).fit()
		
		if len(file_baseline) >= 1:
			assert(np.allclose(df_baseline_human.select_dtypes(numerics).values, df_baseline_human_precomputed.select_dtypes(numerics).values))

		additional_predictors = ['log_topic_variability','log_document_frequency',
								 'log_orthographic_neighborhood_size', 'log_phonological_neighborhood_size',
								'concreteness',  'imageability', 'familiarity','valence','arousal',
								 'log_subtlex_frequency', 'log_subtlex_cd', 'glove_distinctiveness', 'google_ngram_frequency',
								 ]

		# additional_predictors = ['concreteness',  'imageability', 'familiarity','valence','arousal',
		# 						 'log_subtlex_frequency', 'log_subtlex_cd', 'glove_distinctiveness', 'google_ngram_frequency',
		# 						 'log_topic_variability','log_document_frequency',
		# 						 'log_orthographic_neighborhood_size', 'log_phonological_neighborhood_size']

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
		df_additional_predictors_cv_concat = pd.concat(lst_additional_predictors_cv)
		df_additional_predictors_cv_concat = pd.concat([df_baseline_human, df_additional_predictors_cv_concat])
		
		### ANOVA COMPARISON WITH BASELINE ###
		df_additional_predictors_comp_anova = pd.concat(lst_additional_predictors_comp_anova)
		
		# Create 'comparison_index' column (two rows per comparison, so repeat the index twice)
		df_additional_predictors_comp_anova['comparison_index'] = (np.repeat(np.arange(len(df_additional_predictors_comp_anova) / 2), 2))
		
		# Reorganize columns: comparison_index, model, model_add_predictor, F, Pr(>F), ss_diff, df_diff, ssr
		df_additional_predictors_comp_anova = df_additional_predictors_comp_anova[
			['comparison_index', 'model', 'model_add_predictor', 'F', 'Pr(>F)', 'ss_diff', 'df_diff', 'ssr']]
		
		if save:
			df_additional_predictors_cv_concat.to_csv(
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

		# Run within train and test splits (CV)
		df_stepwise, included_features_across_splits = \
			get_cv_score_w_stepwise_regression(df=df, acc1=acc1, acc2=acc2, save=save, result_dir=RESULTDIR,
											   model_name='stepwise',
											   predictors=all_predictors_renamed,
											   save_subfolder=save_subfolder)
		
		# Analyze the most frequently occurring models
		df_stepwise_most_freq = most_frequent_models(included_features_across_splits=included_features_across_splits,
													 num_models_to_report=None) # report all models
		
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
		features_full, pvalues_full = stepwise_selection(X=X_full_demean,
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
				f'across-models_df_cv_NAME-baseline-human-topic_'
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

		## Show frequency distribution of data
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.hist(df['log_subtlex_frequency'], bins=20, color='black')
		ax.set_xlabel('log(SubtLex) frequency')
		ax.set_ylabel('Count')
		ax.set_title('Distribution of log(SubtLex) frequency')
		plt.tight_layout()
		plt.show()

		# Divide into 3 bins based on frequency with equal number of items in each bin
		df['log_subtlex_frequency_bin'] = pd.qcut(df.rank(method='first')['log_subtlex_frequency'], 3,
												  labels=['low', 'medium', 'high'])


		# Get count of number of items in each bin and the mean and standard deviation of the frequency
		df_freq_bin = df.groupby('log_subtlex_frequency_bin').agg(
			{'log_subtlex_frequency': ['count', 'mean', 'std']})
		print(df_freq_bin)

		# Get count of number of items in each bin and the mean and standard deviation of the number of meanings
		df_meanings_bin = df.groupby('log_subtlex_frequency_bin').agg(
			{'num_meanings_human': ['count', 'mean', 'std']})
		print(df_meanings_bin)

		# Get count of number of items in each bin and the mean and standard deviation of the number of synonyms
		df_synonyms_bin = df.groupby('log_subtlex_frequency_bin').agg(
			{'num_synonyms_human': ['count', 'mean', 'std']})
		print(df_synonyms_bin)

		# Run CV regression on data from each bin
		# First, create the new dataframes for each bin (we need to make sure the accs are also dropped in the same way)

		# Create three columns that have 1 if the item is in that bin, and 0 otherwise
		df['log_subtlex_frequency_bin_low'] = np.where(df['log_subtlex_frequency_bin'] == 'low', 1, np.nan)
		df['log_subtlex_frequency_bin_medium'] = np.where(df['log_subtlex_frequency_bin'] == 'medium', 1, np.nan)
		df['log_subtlex_frequency_bin_high'] = np.where(df['log_subtlex_frequency_bin'] == 'high', 1, np.nan)


		# The following will generate dfs with values only for the low, medium, and high bins
		df_low, acc1_low, acc2_low, nan_info_low = \
			drop_nans_from_df(df=df, acc1=acc1, acc2=acc2,
							  predictors=['log_subtlex_frequency_bin_low'])

		df_medium, acc1_medium, acc2_medium, nan_info_medium = \
			drop_nans_from_df(df=df, acc1=acc1, acc2=acc2,
							  predictors=['log_subtlex_frequency_bin_medium'])

		df_high, acc1_high, acc2_high, nan_info_high = \
			drop_nans_from_df(df=df, acc1=acc1, acc2=acc2,
							  predictors=['log_subtlex_frequency_bin_high'])

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

		## Frequency
		# Low
		df_freq_low = get_cv_score(df=df_low,
								   acc1=acc1_low, acc2=acc2_low,
								   save=save, result_dir=RESULTDIR,
								   model_name='freq_low_freq_bin',
								   predictors=['log_subtlex_frequency'], save_subfolder=save_subfolder)

		# Medium
		df_freq_medium = get_cv_score(df=df_medium,
									  acc1=acc1_medium, acc2=acc2_medium,
									  save=save, result_dir=RESULTDIR,
									  model_name='freq_medium_freq_bin',
									  predictors=['log_subtlex_frequency'], save_subfolder=save_subfolder)

		# High
		df_freq_high = get_cv_score(df=df_high,
									acc1=acc1_high, acc2=acc2_high,
									save=save, result_dir=RESULTDIR,
									model_name='freq_high_freq_bin',
									predictors=['log_subtlex_frequency'], save_subfolder=save_subfolder)

		# Merge all CV dfs together
		df_freq_bins_concat = pd.concat([df_human_baseline_low, df_human_baseline_medium, df_human_baseline_high,
										 df_freq_low, df_freq_medium, df_freq_high])

		if save:
			df_freq_bins_concat.to_csv(
				f'{RESULTDIR}/{save_subfolder}/'
				f'cv_summary_preds/'
				f'across-models_df_cv_NAME-freq-bins_'
				f'demeanx-True_demeany-True_permute-False_{date_tag}.csv')


		#















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

			

