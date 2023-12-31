import quantify_meaning_reliability
from paths import *

# # EXPT 1 SYNONYMS
# EXPTFILENAME = 'exp1_data_with_norms_reordered_20221029.csv'
# NORMDATADIR = EXPT1_NORMDIR
# SUBJECTSPLITDIR = EXPT1_SUBJECT_SPLITDIR
# expt = 1
# norm = 'syn'
#
# quantify_meaning_reliability.main(['--EXPTFILENAME', EXPTFILENAME,
#                                          '--NORMDATADIR', NORMDATADIR,
#                                          '--SUBJECTSPLITDIR', SUBJECTSPLITDIR,
#                                          '--expt', str(expt),
#                                          '--norm', norm,])

# EXPT 1 MEANING
# EXPTFILENAME = 'exp1_data_with_norms_reordered_20221029.csv'
# NORMDATADIR = EXPT1_NORMDIR
# SUBJECTSPLITDIR = EXPT1_SUBJECT_SPLITDIR
# expt = 1
# norm = 'meaning'
#
# quantify_meaning_reliability.main(['--EXPTFILENAME', EXPTFILENAME,
#                                          '--NORMDATADIR', NORMDATADIR,
#                                          '--SUBJECTSPLITDIR', SUBJECTSPLITDIR,
#                                          '--expt', str(expt),
#                                          '--norm', norm,])

# # EXPT 2 SYNONYMS
# EXPTFILENAME = 'exp2_data_with_norms_reordered_20221029.csv'
# NORMDATADIR = EXPT2_NORMDIR
# SUBJECTSPLITDIR = EXPT2_SUBJECT_SPLITDIR
# expt = 2
# norm = 'syn'
#
# quantify_meaning_reliability.main(['--EXPTFILENAME', EXPTFILENAME,
#                                             '--NORMDATADIR', NORMDATADIR,
#                                             '--SUBJECTSPLITDIR', SUBJECTSPLITDIR,
#                                             '--expt', str(expt),
#                                             '--norm', norm,])
#


# EXPT 2 MEANING
EXPTFILENAME = 'exp2_data_with_norms_reordered_20221029.csv'
NORMDATADIR = EXPT2_NORMDIR
SUBJECTSPLITDIR = EXPT2_SUBJECT_SPLITDIR
expt = 2
norm = 'meaning'

quantify_meaning_reliability.main(['--EXPTFILENAME', EXPTFILENAME,
                                         '--NORMDATADIR', NORMDATADIR,
                                         '--SUBJECTSPLITDIR', SUBJECTSPLITDIR,
                                         '--expt', str(expt),
                                         '--norm', norm,])

