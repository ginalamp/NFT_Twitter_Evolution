# NFT Twitter Evolution 2021

See `Twitter_NFT_Evolution_2021.pdf` for the full report.

# Usage
need to make sure that tweets.zip is extracted, moved to `datain/clean/` and renamed to `largest_community_tweets.jsonl`

## Run BTM R code
Run `BTM_implementation.Rmd` after running the BTM topic modelling cleaning code. Elbow method.

## Run all Python Scripts
Runs all python scripts automatically.
* Precondition: Need BTM R script output (Run R script first).
* Does not run the Elbow Method

```python3 run_scripts.py```

## Individual scripts
Python scripts can be run individually by going to `src/scripts/` and running:

```python3 <file_name.py>```

# Output
Final output generated in `src/dataout/`

Merged graph output for sentiment and frequency can be generated through running the notebooks.