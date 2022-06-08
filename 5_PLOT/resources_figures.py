### DICT RESOURCES ###
rename_dict_expt1 = {'# meanings (human)': 'num_meanings_human',
					 '# synonyms (human)': 'num_synonyms_human',
					 '# meanings (Wordnet)': 'num_meanings_wordnet',
					 '# synonyms (Wordnet)': 'num_synonyms_wordnet',
					 'Arousal': 'arousal',
					 'Concreteness': 'concreteness',
					 'Familiarity': 'familiarity',
					 'Imageability': 'imageability',
					 'Valence': 'valence',
					  'Log Subtlex frequency': 'log_subtlex_frequency',
					 'Log Subtlex CD': 'log_subtlex_cd',
					 'GloVe distinctiveness': 'glove_distinctiveness',
					 }

rename_dict_expt2 = {'# meanings (human)': 'num_meanings_human',
					 '# synonyms (human)': 'num_synonyms_human',
					 'Arousal': 'arousal',
					 'Concreteness': 'concreteness',
					 'Familiarity': 'familiarity',
					 'Imageability': 'imageability',
					 'Valence': 'valence',
					 'Google n-gram frequency': 'google_ngram_freq',
					 }

rename_dict_expt1_inv = {v: k for k, v in rename_dict_expt1.items()}
rename_dict_expt2_inv = {v: k for k, v in rename_dict_expt2.items()}
rename_dict_expt1_expt2_inv = {**rename_dict_expt1_inv, **rename_dict_expt2_inv}

d_predictors = {'expt1': ['# synonyms (human)', '# meanings (human)',
						   '# synonyms (Wordnet)', '# meanings (Wordnet)',
					  '		Concreteness', 'Imageability', 'Familiarity', 'Valence', 'Arousal',
							'Log Subtlex frequency', 'Log Subtlex CD',
						   'GloVe distinctiveness', ],
				'expt2': ['# synonyms (human)', '# meanings (human)',
					  '		Concreteness', 'Imageability', 'Familiarity', 'Valence', 'Arousal',
						   'Google n-gram frequency',]}

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

d_model_colors = {'synonyms_human': '#f6a0a0',
				  'meanings_human': '#b2ccff',
				  'baseline_human': '#d9baf5',
				  'synonyms_wordnet': '#d94343',
				  'meanings_wordnet': '#477cd4',
				  'baseline_corpus': '#9d52da',}

d_acc_metrics_names = {'acc': 'Accuracy',
					   'hit.rate': 'Hit rate',
					   'false.alarm.rate': 'False alarm rate',}

### LIST RESOURCES ###
order_additional_predictor_models = ['baseline_human',
									 'baseline_human_concreteness', 'baseline_human_imageability',
									 'baseline_human_familiarity',  'baseline_human_valence', 'baseline_human_arousal',
									 'baseline_human_log_subtlex_frequency', 'baseline_human_log_subtlex_cd',
									 'baseline_human_glove_distinctiveness', 'baseline_human_google_ngram_freq']

order_predictors = ['# synonyms (human)', '# meanings (human)',
					'# synonyms (Wordnet)', '# meanings (Wordnet)',
					'Concreteness', 'Imageability', 'Familiarity', 'Valence', 'Arousal',
					'Log Subtlex frequency', 'Log Subtlex CD',
					'GloVe distinctiveness', 'Google n-gram frequency']
