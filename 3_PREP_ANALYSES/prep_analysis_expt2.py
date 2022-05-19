import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

"""
Load expt2 data and rename/reorder the columns.
"""


## SETTINGS ##
save = True

fname = "../1_GET_DATA/exp2_data_with_norms.csv"
fname_accs1 = "../expt2_subject_splits/accs1.csv"
fname_accs2 = "../expt2_subject_splits/accs2.csv"
fname_word_order = "../expt2_subject_splits/word_order_pos.txt" # take POS into account

if __name__ == '__main__':

	## Load data ##
	d = pd.read_csv(fname)
	
	# Rename columns
	d = d.copy().drop(columns=['WORD', 'log_word_len','Word', 'X','Experiment.y','log_freqs'])
	
	d.rename(columns={'word':'word_upper',
					   'SyntCat':'syntactic_category', 'Category':'semantic_category',
					  'NgramFreqArticle':'Google n-gram frequency','NgramFreqNoArticle':'Google n-gram frequency no article',
					  'Input.trial_':'input_trial', 'Experiment.x':'experiment_date',
					  'meanings':'# meanings (human)', 'meanings_n': '# subjects (human) meanings',
					  'num_synonyms':'# synonyms (human)', 'syn_n':'# subjects (human) synonyms',
					  'valence': 'Valence', 'image': 'Imageability', 'fam': 'Familiarity',
					  'concrete': 'Concreteness', 'arousal': 'Arousal',
					  'num_syn':'# synonyms (human) + 1',
					  'w.given.r': '1/# synonyms (human)', 'r.given.w': '1/# meanings (human)',
					  'Acc':'acc', 'Article':'article',
					  'false_alarm_rate':'false.alarm.rate','correct_rejections':'correct.rejections',
					  'false_alarms':'false.alarms', 'hit_rate':'hit.rate'}, inplace=True)
	
	# Create word lower
	d['word_lower'] = d['word_upper'].str.lower()
	
	# Drop five words that were not available in the google ngram frequency database
	nan_index = d[d['Google n-gram frequency'].isnull()].index
	d = d.drop(index=nan_index)
	
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
	
	# Create a column with the word without the article
	print(d.article.unique())
	d['word_no_article_upper'] = d['word_upper']
	d.loc[d['article'] == 'a', 'word_no_article_upper'] = d['word_no_article_upper'].str[2:]
	d.loc[d['article'] == 'an', 'word_no_article_upper'] = d['word_no_article_upper'].str[3:]
	d.loc[d['article'] == 'the', 'word_no_article_upper'] = d['word_no_article_upper'].str[4:]
	d.loc[d['article'] == 'to', 'word_no_article_upper'] = d['word_no_article_upper'].str[3:]
	
	d['word_no_article_lower'] = d['word_no_article_upper'].str.lower()
	
	# Reorganize cols
	cols = d.columns.tolist()
	cols_new_order = ['word_upper', 'word_lower', 'syntactic_category', 'semantic_category',
					  'acc', 'hits', 'misses', 'correct.rejections', 'false.alarms',
					  '# meanings (human)', '# synonyms (human)',
					  'Google n-gram frequency', 'Google n-gram frequency no article',
					  'Arousal', 'Concreteness', 'Familiarity', 'Imageability', 'Valence',
					  '# subjects (human) meanings', '1/# meanings (human)',
					  '# subjects (human) synonyms', '# synonyms (human) + 1', '1/# synonyms (human)',
					  'experiment_date', 'input_trial', 'article', 'word_no_article_upper', 'word_no_article_lower',
					  'multi_word', 'false.alarm.rate', 'hit.rate']
	
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
		d.to_csv(f"exp2_data_with_norms_reordered_gt_{date_tag}.csv")
	
	if save:
		acc1.to_csv(f'../expt2_subject_splits/exp2_accs1_{date_tag}.csv') # with removed and reordered words
		acc2.to_csv(f'../expt2_subject_splits/exp2_accs2_{date_tag}.csv') # with removed and reordered words
		
	
