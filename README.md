# Twitter Crypto Intra Community Analysis
ISMTwitter Crypto Intra Community Analysis - ISMHons 2021

[Overleaf Intra-Community Analysis](https://www.overleaf.com/project/60be345cbd71c045f451e5d1)


## Usage

### Run all
Runs all python scripts (`clean_corpus.py, topic_modelling.py, sentiment_analysis.py`) automatically.

```python3 run_scripts.py```

NOTE: need to run BTM scipt on cleaned corpus first - currently the K20 files for the 100k tweets are
explicitly added to `datain/topic_modelling`, but if a new corpus is used, we need to run the BTM
algorithm on that corpus and add the output to this folder accordingly. (TODO automate this part too)


### Individual scripts
Can run any of the python scripts individually through

```python3 <file_name.py>```

## Output
Usable output is generated in `dataout/`

## Jupyter Notebook
In `notebooks/`

NOTE: the scripts are more up to date than the Jupyter Notebooks - the notebooks were generally used for creating the scripts and being able to follow the state of the dataframes step by step.

Automated pipeline through Jupyter Notebook:
1. clean_corpus.ipynb
2. topic_modelling.ipynb
3. sentiment_analysis.ipynb
