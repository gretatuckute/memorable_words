import os
from os.path import join

###### MEANINGS #######
EXPT1_NORMDIR = '/Users/gt/Documents/GitHub/massive_word_mem/1_GET_DATA_CLEAN/synonym_meaning_norms/SCRIPTS_FILES_FOR_SUBJECT_LEVEL_EXTRACTION/allrawdata_tedmeaningnorms/'
EXPT2_NORMDIR = '/Users/gt/Documents/GitHub/massive_word_mem/1_GET_DATA_CLEAN/synonym_meaning_norms/SCRIPTS_FILES_FOR_SUBJECT_LEVEL_EXTRACTION/evwords_meaning_norms/'


##### SYNONYMS #######


# Define DATADIR as the directory one above this script
DATADIR = os.path.abspath(join(os.path.dirname( __file__ ), '../..'))

# For storing subject splits
EXPT1_SUBJECT_SPLITDIR = join(os.path.dirname( __file__ ), 'expt1_subject_splits')
EXPT2_SUBJECT_SPLITDIR = join(os.path.dirname( __file__ ), 'expt2_subject_splits')

# For storing the reliability results
RELIABILITYDIR = join(os.path.dirname( __file__ ), 'reliability_results')
