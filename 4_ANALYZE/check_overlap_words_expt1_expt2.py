from utils import *

save = True

fname_expt1 = "../3_PREP_ANALYSES/exp1_data_with_norms_reordered_gt_20220519.csv"
fname_expt2 = "../3_PREP_ANALYSES/exp2_data_with_norms_reordered_gt_20220519.csv"

## Load data ##
d = pd.read_csv(fname_expt1).drop(columns=['Unnamed: 0'])
d2 = pd.read_csv(fname_expt2).drop(columns=['Unnamed: 0'])

### Compare distributions of accuracy metrics ###
save_subfolder = 'distributions'
if not os.path.exists(f'../expt1_expt2_results/{save_subfolder}'):
	os.makedirs(f'../expt1_expt2_results/{save_subfolder}')
	print(f'Created directory: {save_subfolder}')
	
	
for metric in ['acc', 'hit.rate', 'false.alarm.rate']:
	
	fig, ax = plt.subplots(1, 1, figsize=(6, 4))
	ax.hist(d[metric], bins=20, label='Experiment 1', alpha=0.5)
	ax.hist(d2[metric], bins=20, label='Experiment 2', alpha=0.5)
	ax.set_xlabel(metric)
	ax.set_ylabel('Count')
	ax.legend()
	fig.tight_layout()
	fig.title = metric
	fig.show()
	
	if save:
		fig.savefig(f'../expt1_expt2_results/{save_subfolder}/'
					f'{metric}_distrib_expt1_expt2_{date_tag}.png', dpi=300)
		fig.savefig(f'../expt1_expt2_results/{save_subfolder}/'
					f'{metric}_distrib_expt1_expt2_{date_tag}.svg', dpi=300)

### Check for overlapping words between d and d2 ###
save_subfolder = 'overlapping_words'
if not os.path.exists(f'../expt1_expt2_results/{save_subfolder}'):
	os.makedirs(f'../expt1_expt2_results/{save_subfolder}')
	print(f'Created directory: {save_subfolder}')
	
	
expt1_expt2_overlap = np.intersect1d(d.word_lower.unique(), d2.word_lower.unique())

# How correlated are their accuracies. Only use expt1_expt2_overlap words
d_overlap = d[d.word_lower.isin(expt1_expt2_overlap)]
d2_overlap = d2[d2.word_lower.isin(expt1_expt2_overlap)]
assert (d_overlap.word_lower.values == d2_overlap.word_lower.values).all()

cols_to_compute_corr_for = ['acc',
							'hit.rate', 'false.alarm.rate',
							'# meanings (human)', '# synonyms (human)',
							'Arousal', 'Concreteness', 'Familiarity', 'Imageability', 'Valence']

lst_r = []
lst_p = []

for col in cols_to_compute_corr_for:
	r, p = pearsonr(d_overlap[col], d2_overlap[col])
	print(f'{col}: r={r:.3f}, p={p:.3f}')
	lst_r.append(r)
	lst_p.append(p)

df_out = pd.DataFrame({'r': lst_r, 'p': lst_p}, index=cols_to_compute_corr_for)

if save:
	fname_out = f"../expt1_expt2_results/" \
				f"{save_subfolder}/" \
				f"overlap_words_expt1_expt2_corr_{date_tag}.csv"
	df_out.to_csv(fname_out)

# Plot correlation for accuracies as a scatter plot with words annotated
for col in cols_to_compute_corr_for:
	for annotate in [True, False]:
		annotate_str = ['_annotated' if annotate else ''][0]
		
		plt.figure(figsize=(8, 8))
		plt.scatter(d_overlap[col].values, d2_overlap[col], s=150, alpha=0.3, linewidths=0, color='black')
		if annotate:
			for i, word in enumerate(d_overlap.word_lower.unique()):
				plt.annotate(word, (d_overlap[col].values[i], d2_overlap[col].values[i]), fontsize=15)
		plt.xlabel(f'Experiment 1', fontsize=18)
		plt.ylabel(f'Experiment 2', fontsize=18)
		plt.title(f'Experiement 1 vs. Experiment 2 for overlapping words:\n{col}', fontsize=18)
		if col == 'acc':
			plt.xlim(0.6, 1)
			plt.ylim(0.6, 1)
		# Add r-value from df_out of 'acc'
		plt.text(0.93, 0.97, f'r={df_out.loc[col, "r"]:.2f}', fontsize=20)
		# Decrease number of ticks and make bigger
		if col == 'acc':
			plt.xticks(np.arange(0.6, 1.01, 0.1), fontsize=18)
			plt.yticks(np.arange(0.6, 1.01, 0.1), fontsize=18)
		plt.tight_layout()
		if save:
			plt.savefig(f'../expt1_expt2_results/{save_subfolder}/'
						f'{col}_corr_scatter_expt1_expt2{annotate_str}_{date_tag}.png', dpi=300)
			plt.savefig(f'../expt1_expt2_results/{save_subfolder}/'
						f'{col}_corr_scatter_expt1_expt2{annotate_str}_{date_tag}.svg', dpi=300)
		plt.show()
