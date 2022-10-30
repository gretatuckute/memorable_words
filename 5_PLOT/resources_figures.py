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
					 'Google n-gram frequency': 'google_ngram_frequency',
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
					 'Google n-gram frequency': 'google_ngram_frequency',
					 }

rename_dict_expt1_inv = {v: k for k, v in rename_dict_expt1.items()}
rename_dict_expt2_inv = {v: k for k, v in rename_dict_expt2.items()}
rename_dict_expt1_expt2_inv = {**rename_dict_expt1_inv, **rename_dict_expt2_inv}

d_predictors = {'expt1': ['# synonyms (human)', '# meanings (human)',
						   '# synonyms (Wordnet)', '# meanings (Wordnet)',
					  'Concreteness', 'Imageability', 'Familiarity', 'Valence', 'Arousal',
						'Google n-gram frequency',
						'Log Subtlex frequency', 'Log Subtlex CD',
						'GloVe distinctiveness', ],
				'expt2': ['# synonyms (human)', '# meanings (human)',
						'Google n-gram frequency',
					  'Concreteness', 'Imageability', 'Familiarity', 'Valence', 'Arousal',
						]}

d_model_names = {'synonyms_human': '# synonyms',
				 'meanings_human': '# meanings',
				 'baseline_human': '# synonyms and # meanings',
				 'baseline_human_no_nans_topic': '# synonyms and # meanings (topic limited)',
				 'baseline_human_no_nans_orth_phon': '# synonyms and # meanings (orth limited)',
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
				'baseline_human_google_ngram_frequency': 'Google n-gram frequency',
				 'CD': 'Log Subtlex CD',
				 'CD_no_nans_topic': 'Log Subtlex CD (topic limited)',
				 'topicvar': 'Log TASA TV',
				'orth_neighborhood': 'Log orthographic neighborhood',
				'phon_neighborhood': 'Log phonological neighborhood',
				'baseline_human_low_freq_bin': 'Low frequency bin',
				'baseline_human_medium_freq_bin': 'Mid frequency bin',
				'baseline_human_high_freq_bin': 'High frequency bin',
}

d_model_colors = {'synonyms_human': '#f6a0a0',
				  'meanings_human': '#b2ccff',
				  'baseline_human': '#d9baf5',
					'baseline_human_no_nans_topic': '#d9baf5',
					'baseline_human_no_nans_orth_phon': '#d9baf5',
				  'synonyms_wordnet': '#d94343',
				  'meanings_wordnet': '#477cd4',
				  'baseline_corpus': '#9d52da',
				  'CD': '#63D667',
				  'CD_no_nans_topic': '#63D667',
				  'topicvar': '#B6E8B8',
				  'orth_neighborhood': '#C3C191',
				  'phon_neighborhood': '#E4E2BB',
				  'baseline_human_low_freq_bin': '#AB91E4',
				  'baseline_human_medium_freq_bin': '#8254E8',
				  'baseline_human_high_freq_bin': '#5511EA',
				  }

d_acc_metrics_names = {'acc': 'Accuracy',
					   'hit.rate': 'Hit rate',
					   'false.alarm.rate': 'False alarm rate',}

d_expt_names = {'expt1': 'Expt 1',
				'expt2': 'Expt 2',}

d_model_names_anova = {'acc_demean ~ num_synonyms_human_demean': 'Accuracy ~ Num_Synonyms',
				 'acc_demean ~ num_meanings_human_demean': 'Accuracy ~ Num_Meanings',
				  'acc_demean ~ num_synonyms_human_demean + num_meanings_human_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings',
				 'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + concreteness_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Concreteness',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + imageability_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Imageability',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + familiarity_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Familiarity',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + valence_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Valence',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + arousal_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Arousal',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + google_ngram_frequency_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Google n-gram frequency',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + log_subtlex_frequency_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Log Subtlex frequency',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + log_subtlex_cd_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + Log Subtlex CD',
					   'acc_demean ~ num_meanings_human_demean + num_synonyms_human_demean + glove_distinctiveness_demean': 'Accuracy ~ Num_Synonyms + Num_Meanings + GloVe distinctiveness',
					   }

### LIST RESOURCES ###
order_additional_predictor_models = ['baseline_human',
									 'baseline_human_concreteness', 'baseline_human_imageability',
									 'baseline_human_familiarity',  'baseline_human_valence', 'baseline_human_arousal',
									  'baseline_human_google_ngram_frequency',
									 'baseline_human_log_subtlex_frequency', 'baseline_human_log_subtlex_cd',
									 'baseline_human_glove_distinctiveness', ]

order_predictors = ['# synonyms (human)', '# meanings (human)',
					'# synonyms (Wordnet)', '# meanings (Wordnet)',
					'Concreteness', 'Imageability', 'Familiarity', 'Valence', 'Arousal',
					'Google n-gram frequency',
					'Log Subtlex frequency', 'Log Subtlex CD',
					'GloVe distinctiveness', ]
