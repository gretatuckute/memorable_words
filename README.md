# Intrinsically memorable words have unique associations with their meanings

Data and code accompanying the preprint: Tuckute*, G., Mahowald*, K., Isola, P., Gibson, E., Oliva, A., & Fedorenko, E. (2022). Intrinsically memorable words have unique associations with their meanings. https://doi.org/10.31234/osf.io/p6kv9

What makes a word memorable? We collected word recognition accuracy for n=2,109 (Experiment 1) and n=2,165 (Experiment 2) words from more than 600 participants in each experiment. This repository makes the memorability data publicly available and provides code to investigate which features are predictive of memorability.

## Environment
The environment does not require any sophisticated packages and would run in any Python 3 environment with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/), [statsmodels](https://www.statsmodels.org/stable/index.html), [seaborn](https://seaborn.pydata.org/) and [matplotlib](https://matplotlib.org/). However, to use the exact Python 3.8.13 environment used in the paper, install it as:

```
conda env create -f environment.yml
```

## Data availability
The word memorability data is available in /3_PREP_ANALYSES/:

Experiment 1: [*exp1_data_with_norms_reordered_20220708.csv*](https://github.com/gretatuckute/memorable_words/blob/main/3_PREP_ANALYSES/exp1_data_with_norms_reordered_20220708.csv) (2,109 rows according to experimental items as denoted in the columns *word_upper*. The *acc* column contains the word recognition accuracy. Additional columns contain metadata and feature norms)

Experiment 2: [*exp2_data_with_norms_reordered_20220708.csv*](https://github.com/gretatuckute/memorable_words/blob/main/3_PREP_ANALYSES/exp2_data_with_norms_reordered_20220708.csv) (2,165 rows according to experimental items as denoted in the columns *word_upper*. The *acc* column contains the word recognition accuracy. Additional columns contain metadata and feature norms)

## Analyses
To analyze respectively Experiment 1 and Experiment 2, run [/4_ANALYZE/analyze_expt1.py](https://github.com/gretatuckute/memorable_words/blob/main/4_ANALYZE/analyze_expt1.py) and [/4_ANALYZE/analyze_expt2.py](https://github.com/gretatuckute/memorable_words/blob/main/4_ANALYZE/analyze_expt2.py).
By default, all analyses reported in the preprint will be performed. To only run certain analyses, change the settings [in the beginning of the script](https://github.com/gretatuckute/memorable_words/blob/main/4_ANALYZE/analyze_expt1.py#L3). 

## Figures and tables
To generate figures and tables for Experiment 1 and Experiment 2, run [/5_PLOT/generate_figures.py](https://github.com/gretatuckute/memorable_words/blob/main/5_PLOT/generate_figures.py) and [/5_PLOT/generate_tables.py](https://github.com/gretatuckute/memorable_words/blob/main/5_PLOT/generate_tables.py). These scripts assume that all analyses in 4_ANALYZE have been performed. The date on which the analyses were performed can be inputted [here](https://github.com/gretatuckute/memorable_words/blob/main/5_PLOT/generate_figures.py#L7). 

## Citation
If you found this repository or data useful, please cite:

```
@ARTICLE{TuckuteMahowald2022,
   author = {{Tuckute*}, G. and
              {Mahowald*}, K. and
              {Isola}, P. and
              {Oliva}, A. and
              {Gibson}, E. and
              {Fedorenko}, E.},
    title = "{Intrinsically memorable words have unique associations with their meanings}",
  journal = {Psyarxiv},
     year = 2022,
    doi = https://doi.org/10.31234/osf.io/p6kv9
}
```
