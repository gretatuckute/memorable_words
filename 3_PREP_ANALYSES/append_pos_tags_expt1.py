import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
now = datetime.now()
date_tag = now.strftime("%Y%m%d")

"""
Load the output for prep_analysis_expt1.py and append POS tags post-hoc.
POS tags live in /expt1_pos_tags/pos_exp1_info.csv
"""

## SETTINGS ##
save = True

fname = "exp1_data_with_norms_reordered_20221029.csv"

if __name__ == '__main__':

	## Load data ##
	d = pd.read_csv(fname)

	d_pos = pd.read_csv('expt1_pos_tags/pos_exp1_info.csv')
	# Create col "word_upper" from Word
	d_pos['word_upper'] = d_pos['Word'].str.upper()

	# Get d_pos for the words that exist in d "word_upper"
	d_pos = d_pos[d_pos['word_upper'].isin(d['word_upper'])]

	# Index in the same order as d
	d_pos = d_pos.set_index('word_upper')

	# Append POS tags to d
	d_merge = d.join(d_pos, on='word_upper', how='left')

	# Save the original d with suffix _before_pos
	if save:
		d.to_csv(f"{fname[:-4]}_before_pos.csv", index=False)

	# Save the merged d with the POS tags, original name
	if save:
		d_merge.to_csv(fname, index=False)
