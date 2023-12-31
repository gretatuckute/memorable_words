import pandas as pd
import numpy as np
import os
from os.path import join
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import argparse

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        print(f'Already a boolean: {v}')
        return v
    if v.lower() in ('true', 't'):
        print(f'String arg - True: {v}')
        return True
    elif v.lower() in ('false', 'f'):
        print(f'String arg - False: {v}')
        return False
    else:
        print(f'String arg - {v}')
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    """If string is 'None', return None. Else, return the string"""
    if v is None:
        print(f'Already None: {v}')
        return v
    if v.lower() in ('none'):
        print(f'String arg - None: {v}')
        return None
    else:
        return v
def split_half(df: pd.DataFrame,
               random_state: int = 0,
               item_col: str = 'Input.trial_',
               participant_col: str = 'Participant') -> (pd.DataFrame, pd.DataFrame):
    """
    For each item, we want to split the participants into two groups
    So, for each item, we get the unique values in participant_col and split them into two groups.
    To allow for different splits, we shuffle the unique values in participant_col before splitting them into two groups.

    :param df: pd.DataFrame (rows are trials, columns are variables)
    :param item_col: str (name of column that contains the items)
    :param participant_col: str (name of column that contains the participants)

    """
    # For each item, we want to split the participants into two groups
    # Get the unique values in col
    lst_df1 = []
    lst_df2 = []
    for item in df[item_col].unique():
        # Take half of the participants and put them in one df
        df_item = df[df['Input.trial_'] == item]
        # Get the unique values in col
        unique_vals = df_item[participant_col].unique()
        # Shuffle the unique values
        np.random.seed(random_state)
        np.random.shuffle(unique_vals)
        # Split the unique values in col into two groups
        unique_vals1 = unique_vals[:len(unique_vals) // 2]
        unique_vals2 = unique_vals[len(unique_vals) // 2:]
        assert len(unique_vals1) + len(unique_vals2) == len(unique_vals)
        assert np.intersect1d(unique_vals1, unique_vals2).size == 0
        # Put the participants in df1 and df2
        df1 = df_item[df_item[participant_col].isin(unique_vals1)]
        df2 = df_item[df_item[participant_col].isin(unique_vals2)]
        lst_df1.append(df1)
        lst_df2.append(df2)

    # Concatenate the dfs
    df1 = pd.concat(lst_df1, axis=0)
    df2 = pd.concat(lst_df2, axis=0)

    return df1, df2