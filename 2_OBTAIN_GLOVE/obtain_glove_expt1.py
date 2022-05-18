import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

## SETTINGS ##
save = True
plot = True

fname = "../1_GET_DATA/exp1_data_with_norms.csv"
GLOVEDIR = '/Users/gt/Dropbox (MIT)/SemComp/features/database/glove_database/'


### GloVe Functions ###
def get_vocab(input_list: list):
	"""
	Return set of unique tokens in input list.
	
	Args:
		input_list (list): list of lists of tokens (strings)
	
	Returns:
		vocab (set): set of unique tokens in input list
	"""
	
	vocab = set()
	for sent in input_list:
		for token in sent:
			vocab.add(token)
	return vocab


def read_glove_embed(vocab: set,
					 glove_path: str,):
	"""
	Read through the GloVe embedding file to find embeddings for target words in vocab.
	
	Args:
		vocab (set): set of words to find embeddings for
		glove_path (str): path to GloVe embedding file
		
	Returns:
		w2v (dict): dictionary of word:vector pairs
	
	"""
	
	w2v = {}
	with open(glove_path, 'r') as f:
		for line in tqdm(f):
			tokens = line.strip().split(' ')
			w = tokens[0]  # the word
			if w in vocab:
				v = tokens[1:]  # first index is the word
				w2v[w] = np.array(v, dtype=float)
	return w2v


def get_glove_embed(input_list: list,
					w2v: dict,):
	"""
	Fetches glove embeddings for a list of strings.
	If multiple words are present in each list: averages the embedding,
	if just a single-word is present in the list: fetches the single embedding.
	
	Args:
		input_list (list): list of lists of tokens (strings: words/sentences)
		w2v (dict): dictionary of word:vector pairs
		
	Returns:
		glove_embed (list): list of lists of glove embeddings
		contains_na (list): list of booleans, True if word not in w2v
		NA_words (list): list of words not in w2v

	"""
	
	glove_embed = []
	NA_words = set()
	contains_na = []
	for sent_idx, sent in enumerate(input_list):
		sent_embed = []
		flag_na = False
		if sent_idx % 200 == 0:
			print(sent_idx, flush=True)
		
		# find multi-word words
		split_sent = sent.split(' ')  # contains words
		print(f'Sentence: {split_sent}')
		
		for token_idx, token in enumerate(split_sent):
			print(f'Tokens: {token}')
			if token in w2v:
				sent_embed.append(w2v[token])
			else:
				NA_words.add(token)
				flag_na = True
				a = np.empty((300))
				a[:] = np.nan
				sent_embed.append(a)
				print(f'Added NaN emb for {token}')
		
		#         print(f'Sentence embedding: {sent_embed}')
		glove_embed.append(np.nanmean(np.array(sent_embed), axis=0))
		contains_na.append(flag_na)
	
	try:
		print(f'Number of words with NA glove embedding: {len(NA_words)}')
		print('Example NA words:', list(NA_words))
	except:
		print('No NaN words!')
	
	return glove_embed, contains_na, NA_words


if __name__ == '__main__':
	## Load data ##
	df = pd.read_csv(fname)
	
	# Obtain all unique words
	vocab = set(df.Word.str.lower())
	
	# Get GloVe dict for words of interest
	w2v = read_glove_embed(vocab, GLOVEDIR + 'glove.840B.300d.txt')
	print(
		f'The following {len(vocab - w2v.keys())} words not available in GloVe dict: {vocab - w2v.keys()}')  # missing words
	
	if save:
		w2v_df = pd.DataFrame.from_dict(w2v, orient='index')
		w2v_df.to_pickle(f'w2v_expt1_{date_tag}.pkl')
	
	lst_glove_emb = []
	matrix_glove_emb = np.empty((len(df), 300))
	for row in df.itertuples():
		word_of_interest = row.Word.lower()
		if word_of_interest in w2v:
			glove_embed = w2v[word_of_interest]
			matrix_glove_emb[row.Index] = glove_embed
		else:
			glove_embed = np.empty(300)
			glove_embed[:] = np.nan
		lst_glove_emb.append(glove_embed)
		
	# Matrix glove emb contains the glove embeddings for each word, as sorted in the experiment df.
	glove_sim = cosine_similarity(matrix_glove_emb)
	
	if plot: # visualize glove embedding similarities
		plt.figure(figsize=(20, 20))
		sns.heatmap(glove_sim[:100, :100], square=True, cmap='RdBu_r', center=0)
		plt.xticks(np.arange(100), df.Word.values[:100])
		plt.yticks(np.arange(100), df.Word.values[:100])
		plt.show()
	
	# Obtain GloVe distinctiveness metric
	# A given word's embedding and the average of cosine similarity for all other words (e.g., n=2109-1=2108)
	lst_distinctiveness = []
	for i in range(len(glove_sim)):
		sim_all_words = glove_sim[i, :]
		distinctiveness_metric = np.mean(np.delete(sim_all_words, i))  # delete the corr with yourself (index i)
		lst_distinctiveness.append(distinctiveness_metric)
	
	df['GloVe distinctiveness'] = lst_distinctiveness
	
	df.to_csv("exp1_data_with_norms_w_glove.csv")
	