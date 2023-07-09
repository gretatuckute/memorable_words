from utils import *

save = False

fname_expt1 = "../3_PREP_ANALYSES/exp1_data_with_norms_reordered_20221029.csv"
fname_expt2 = "../3_PREP_ANALYSES/exp2_data_with_norms_reordered_20221029.csv"

## Load data ##
d = pd.read_csv(fname_expt1).drop(columns=['Unnamed: 0'])
d2 = pd.read_csv(fname_expt2).drop(columns=['Unnamed: 0'])

### Compare distributions of accuracy metrics ###
save_subfolder = 'corrs_with_existing_norms'
if not os.path.exists(f'../expt1_expt2_results/{save_subfolder}'):
	os.makedirs(f'../expt1_expt2_results/{save_subfolder}')
	print(f'Created directory: {save_subfolder}')
	
	
#### LOAD VALENCE ANDAROUSAL DATA FROM Mohammad, 2018 ####
fname_val_arousal = '/Users/gt/Documents/GitHub/sent-spaces-comp/database/feature_database/NRC-VAD-Lexicon.xlsx'

val_arousal = pd.read_excel(fname_val_arousal).rename(columns={'Valence': 'Valence_Mohammad', 'Arousal': 'Arousal_Mohammad'})

# Find overlapping words from d and d2 (col: word_lower) and val_arousal (col: Word)
# Expt 1
d_val_arousal = pd.merge(d, val_arousal, how='inner', left_on='word_lower', right_on='Word')
print(f'Expt 1: {d_val_arousal.shape[0]} overlapping words with valence/arousal data out of {d.shape[0]} total words')

# Expt 2
d2_val_arousal = pd.merge(d2, val_arousal, how='inner', left_on='word_lower', right_on='Word')
print(f'Expt 2: {d2_val_arousal.shape[0]} overlapping words with valence/arousal data out of {d2.shape[0]} total words')

# Get correlation between Valence/Arousal and Valence_Mohammad/Arousal_Mohammad
# Expt 1
print(f'Expt 1: Valence: {pearsonr(d_val_arousal["Valence"], d_val_arousal["Valence_Mohammad"])[0]:.2f}')
print(f'Expt 1: Arousal: {pearsonr(d_val_arousal["Arousal"], d_val_arousal["Arousal_Mohammad"])[0]:.2f}')

# Expt 2
print(f'Expt 2: Valence: {pearsonr(d2_val_arousal["Valence"], d2_val_arousal["Valence_Mohammad"])[0]:.2f}')
print(f'Expt 2: Arousal: {pearsonr(d2_val_arousal["Arousal"], d2_val_arousal["Arousal_Mohammad"])[0]:.2f}')

#### LOAD CONCRETENESS DATA FROM Brysbaert, 2014 ####
fname_concreteness = '/Users/gt/Documents/GitHub/sent-spaces-comp/database/feature_database/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx'

val = pd.read_excel(fname_concreteness).rename(columns={'Conc.M': 'Concreteness_Brysbaert'})

# Find overlapping words from d and d2 (col: word_lower) and val_arousal (col: Word)
# Expt 1
d_val = pd.merge(d, val, how='inner', left_on='word_lower', right_on='Word')
print(f'Expt 1: {d_val.shape[0]} overlapping words with concrereteness data out of {d.shape[0]} total words')

# Expt 2
d2_val = pd.merge(d2, val, how='inner', left_on='word_lower', right_on='Word')
print(f'Expt 2: {d2_val.shape[0]} overlapping words with concrereteness data out of {d2.shape[0]} total words')

# Get correlation between Concreteness and Concreteness_Brysbaert
# Expt 1
print(f'Expt 1: Concreteness: {pearsonr(d_val["Concreteness"], d_val["Concreteness_Brysbaert"])[0]:.2f}')

# Expt 2
print(f'Expt 2: Concreteness: {pearsonr(d2_val["Concreteness"], d2_val["Concreteness_Brysbaert"])[0]:.2f}')

