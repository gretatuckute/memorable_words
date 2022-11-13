import numpy as np
import pandas as pd
from datetime import datetime

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

"""
Load expt2 data and rename/reorder the columns.
"""


## SETTINGS ##
save = True

fname = "../1_GET_DATA/exp2_data_with_norms_20221010.csv"
fname_accs1 = "../expt2_subject_splits/accs1.csv"
fname_accs2 = "../expt2_subject_splits/accs2.csv"
fname_word_order = "../expt2_subject_splits/word_order_pos.txt" # take POS into account

if __name__ == '__main__':

	## Load data ##
	d = pd.read_csv(fname)
	
	# Rename columns
	d = d.copy(deep=True).drop(columns=['Word', 'WORD.1', 'subcat4', 'word_upper', 'Word_article'])
	
	d.rename(columns={'num_synonyms': '# synonyms (human)', 'num_meanings': '# meanings (human)',
					  'Lg10CD': 'Log Subtlex CD', 'Lg10WF': 'Log Subtlex frequency',
					  'google_ngram_frequency': 'Google n-gram frequency',
					  'valence': 'Valence', 'image': 'Imageability', 'fam': 'Familiarity',
					  'concrete': 'Concreteness', 'arousal': 'Arousal',
					  'word': 'word_lower', 'WORD': 'word_upper',
					  'Article': 'article',
					  'SyntCat': 'syntactic_category', 'Category': 'category',
					  'subcat1': 'subcategory_1', 'subcat2': 'subcategory_2', 'subcat3': 'subcategory_3',
					  'LgTV_T=1700': 'Log topic variability', 'LgDF': 'Log document frequency',
					  'orth_neighborhood': 'Orthographic neighborhood size',
					  'phon_neighborhood': 'Phonological neighborhood size',
					  'log_orth_neighborhood': 'Log orthographic neighborhood size',
					  'log_phon_neighborhood': 'Log phonological neighborhood size',
					  }, inplace=True)
	
	d['word_lower'] = d['word_lower'].str.lower()
	d['word_lower_no_article'] = d['word_lower_no_article'].str.lower()
	
	# Check how many multi-word vs single word items there were or with dashes
	d_dashes = d[d['word_upper'].str.contains('-')].index
	d_multi = d[d['word_upper'].str.contains(' ')].index
	
	# Get index of all multi-word items and items with dashes
	multi_index = d_multi.union(d_dashes)
	
	# Find the single-words ones (not in multi_index)
	single_index = d[~d.index.isin(multi_index)].index
	
	d['multi_word'] = np.nan
	d.loc[multi_index, 'multi_word'] = 1
	d.loc[single_index, 'multi_word'] = 0
	
	# Reorganize cols
	cols = d.columns.tolist()
	cols_new_order = ['word_upper', 'word_lower', 'word_lower_no_article', 'article',
					  'syntactic_category', 'category', 'subcategory_1', 'subcategory_2', 'subcategory_3',
					  'acc', 'hits', 'misses',
					  'correct_rejections', 'false_alarms', 'false_alarm_rate', 'hit_rate',
					  '# meanings (human)', '# synonyms (human)',
					  'Log Subtlex frequency', 'Log Subtlex CD', 'Google n-gram frequency',
					  'Concreteness', 'Imageability', 'Familiarity', 'Valence', 'Arousal',
					  'num_ratings_meanings', 'num_ratings_synonyms', 'multi_word',
					  'Log topic variability', 'Log document frequency',
					  'Log orthographic neighborhood size', 'Log phonological neighborhood size',
					  'num_ratings_meanings', 'num_ratings_synonyms',
					  ]

	d = d[cols_new_order]
	
	## Prepare accuracies for splits (remove words that were excluded) ##
		
	# Each single row is accuracy for half of the participants, and columns are words
	acc1 = pd.read_csv(fname_accs1, header=None)
	acc2 = pd.read_csv(fname_accs2, header=None)
	
	# order of the words for acc1 and acc2
	accs_word_order = pd.read_csv(fname_word_order, header=None, encoding= 'unicode_escape')
	
	# get the parenthesis with POS into a separate column
	accs_word_order['POS'] = accs_word_order[0].str.extract('\((.*?)\)')
	
	# remove the POS from the word and strip the space
	accs_word_order[0] = accs_word_order[0].str.replace('\(.*?\)', '').str.strip()
	
	# rename to word_lower and lowercase
	accs_word_order.rename(columns={0:'word_normal_case', 1:'POS'}, inplace=True)
	accs_word_order['word_lower'] = accs_word_order['word_normal_case'].str.lower()
	
	assert(len(np.unique(accs_word_order['word_lower']) == 2222))
	assert(len(np.intersect1d(d['word_lower'].values, accs_word_order['word_lower'].values, )) == len(d))
	
	## Add in the word names for acc1 and acc2 based on the accs_word_order ##
	acc1.columns = accs_word_order['word_lower']
	acc2.columns = accs_word_order['word_lower']

	# Exclude the ones that were are not present in d['word_lower'] (57 total) and reorder
	acc1 = acc1[d['word_lower'].values]
	acc2 = acc2[d['word_lower'].values]
	
	assert(len(np.unique(acc1.columns)) == len(np.unique(acc2.columns)))
	assert (np.unique(acc1.columns) == np.unique(d['word_lower'])).all()
	assert (np.unique(acc2.columns) == np.unique(d['word_lower'])).all()
	
	# Print out excluded words
	excluded_words = np.setdiff1d(np.unique(accs_word_order['word_lower']), np.unique(d['word_lower'].values))
	print('Excluded words:')
	print(excluded_words)
	
	if save:
		d.to_csv(f"exp2_data_with_norms_reordered_{date_tag}.csv")
		acc1.to_csv(f'../expt2_subject_splits/exp2_accs1_{date_tag}.csv') # with removed and reordered words
		acc2.to_csv(f'../expt2_subject_splits/exp2_accs2_{date_tag}.csv') # with removed and reordered words
		
	
