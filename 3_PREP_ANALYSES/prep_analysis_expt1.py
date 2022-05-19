import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
now = datetime.now()
date_tag = now.strftime("%Y%m%d")

"""
Load expt1 data (after the GloVe extraction) and rename/reorder the columns.
"""

## SETTINGS ##
save = True

fname = "../2_OBTAIN_GLOVE/exp1_data_with_norms_w_glove.csv"
fname_accs1 = "../expt1_subject_splits/accs1.csv"
fname_accs2 = "../expt1_subject_splits/accs2.csv"
fname_word_order = "../expt1_subject_splits/word_order.txt"

if __name__ == '__main__':

	## Load data ##
	d = pd.read_csv(fname)
	
	# Rename columns
	d = d.copy().drop(columns=['FREQcount', 'CDcount', 'Cdlow', 'X', 'freq', 'M','FREQlow', 'word.y', 'mean.cor'])
	
	d.rename(columns={'num_synonyms': '# synonyms (human)', 'meanings': '# meanings (human)',
									 'wn_num_synonyms': '# synonyms (Wordnet)', 'wn_num_meanings': '# meanings (Wordnet)',
					  				 'Lg10CD': 'Log Subtlex CD', 'Lg10WF': 'Log Subtlex frequency',
									 'valence': 'Valence', 'image': 'Imageability', 'fam': 'Familiarity',
									 'concrete': 'Concreteness', 'arousal': 'Arousal',
					  				 'SUBTLWF': 'Subtlex frequency', 'SUBTLCD': 'Subtlex CD',
									 'syn_n': '# subjects (human) synonyms', 'meanings_n': '# subjects (human) meanings',
					  				'w.given.r': '1/# synonyms (human)', 'r.given.w': '1/# meanings (human)',
					  				'num_syn': '# synonyms (human) + 1',
					                'word.x': 'word_lower', 'Word': 'word_upper',	'Acc':'acc',}, inplace=True)
	cols = d.columns.tolist()
	cols_new_order = ['word_upper', 'word_lower', 'pos', 'acc', 'hits', 'misses', 'correct.rejections', 'false.alarms',
					  '# meanings (human)', '# synonyms (human)', '# meanings (Wordnet)', '# synonyms (Wordnet)',
					  'Subtlex frequency', 'Log Subtlex frequency', 'Subtlex CD', 'Log Subtlex CD',
					  'Arousal', 'Concreteness', 'Familiarity', 'Imageability', 'Valence', 'GloVe distinctiveness',
					  '# subjects (human) meanings', '1/# meanings (human)',
					  '# subjects (human) synonyms', '# synonyms (human) + 1', '1/# synonyms (human)',
					  'false.alarm.rate', 'hit.rate']

	d = d[cols_new_order]
	
	if save:
		d.to_csv(f"exp1_data_with_norms_reordered_gt_{date_tag}.csv")
	
	## Prepare accuracies for splits (remove words that were excluded) ##
		
	# Each single row is accuracy for half of the participants, and columns are words
	acc1 = pd.read_csv(fname_accs1, header=None)
	acc2 = pd.read_csv(fname_accs2, header=None)
	
	# order of the words for acc1 and acc2
	accs_word_order = pd.read_csv(fname_word_order, header=None) #2109
	accs_word_order_str = [x[0] for x in accs_word_order.values]
	
	# Delete words that did not have enough norms (113)
	excluded_words = np.setdiff1d(accs_word_order, d.word_lower)
	indices_to_exclude = [accs_word_order_str.index(x) for x in excluded_words] # get index of words to exclude
	cols_to_exclude = np.asarray([False if x in indices_to_exclude else True for x in np.arange(len(accs_word_order))]) # 0 if exclude
	
	x_arr = np.asarray(accs_word_order_str, dtype=object)
	excluded_words_check = x_arr[~cols_to_exclude]  # Get items
	assert (excluded_words == np.sort(excluded_words_check)).all()
	
	# Now we want to make sure that the words in accs1 and accs2 are the same as in dataframe (d)
	# First exclude the deleted words from the accs_word_order_str
	accs_word_order_str_new = [x for x in accs_word_order_str if x not in excluded_words]
	
	acc1_with_excluded_words = acc1.loc[:, cols_to_exclude]
	acc1_with_excluded_words.columns = accs_word_order_str_new
	
	acc2_with_excluded_words = acc2.loc[:, cols_to_exclude]
	acc2_with_excluded_words.columns = accs_word_order_str_new
	
	# Assert that word order txt file matches with the accs csv files
	
	# Read in word order text file anew to ensure that no variables were manipulated incorrectly
	word_order = pd.read_csv(fname_word_order, header=None)
	word_order_lower = word_order[0].str.lower()
	
	# Assert that word order txt file matches with the accs csv files
	assert (len(np.intersect1d(acc1_with_excluded_words.columns.values, word_order_lower.to_list())) == acc1_with_excluded_words.shape[1])
	assert (len(np.intersect1d(acc1_with_excluded_words.columns.values, accs_word_order_str_new)) == acc1_with_excluded_words.shape[1])
	
	# Next, reorder the words (columns) so that d and the accs files match up
	accs1 = acc1_with_excluded_words[d.word_lower.values]
	accs2 = acc2_with_excluded_words[d.word_lower.values]
	
	assert(len(np.unique(accs1.columns)) == len(np.unique(accs2.columns)))
	assert (np.unique(accs1.columns) == np.unique(d['word_lower'])).all()
	assert (np.unique(accs2.columns) == np.unique(d['word_lower'])).all()
	
	if save:
		accs1.to_csv(f'../expt1_subject_splits/exp1_accs1_{date_tag}.csv') # with removed and reordered words
		accs2.to_csv(f'../expt1_subject_splits/exp1_accs2_{date_tag}.csv') # with removed and reordered words