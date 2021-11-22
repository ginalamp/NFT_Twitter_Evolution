# NFT Twitter Evolution 2021

See `Twitter_NFT_Evolution_2021.pdf` for the full report.

# Usage

## Install dependencies
1. Create a Python virtual environment with `python3 -m venv env`
2. Activate the venv with `source env/bin/activate` (it can be deactivated by running `deactivate` or by exiting the terminal)
3. Install Python dependencies with `pip3 install -r requirements.txt`

## Setup
1. Extract `tweets.zip`, rename the extracted file to `largest_community_tweets.jsonl`, and move the file to `datain/clean/`
2. In `src/scripts/`, run `python3 clean_corpus.py`. This will clean the data.

## Run BTM R code
This is for topic modelling. Currently it assumes that the number of topics is 11 (as this was determined through using the Elbow Method on the LogLik values on the dataset used).

The BTM code is a bit manual currently, but is well documented if you want to automate it a bit into a script.
* Install dependencies according to imports.
* LogLik is set to False by default (set to True if want to run again). 
    - Need to manually take note of the LogLik values (csv with the number of topics and LogLik value is used as input in `python3 elbow_method.py`)

1. Run the `BTM_implementation.Rmd` notebook in `src/BTM_topics`.
2. You can take note of the LogLik values and then run `python3 elbow_method.py` in `src/scripts` if you want to find the most optimal number of topics for your dataset. 

## Run all Python Scripts
Runs all Python scripts automatically and in order.
* Precondition: Need BTM R script output (Run R script first).
* Does not run the Elbow Method.

```python3 run_scripts.py```


## Individual scripts
Python scripts can be run individually by going to `src/scripts/` and running:

```python3 <file_name.py>```

or by importing the modules in a python interpreter, for example in `ipython3`:

```py
import frequency as f
f.run() # runs frequency code with default parameters
f.run(overall=True) # runs frequency code on overall data
f.run(trendline=False) # without trend line
f.run(selected_topic = 3, trendline=False) # for topic 3 and without trend line
```

# Output
Final output generated in `src/dataout/`

Merged graph output for sentiment can be generated through running the `merge-sentiment-graphs.ipynb` notebook.