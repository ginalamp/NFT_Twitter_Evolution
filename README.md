# NFT Twitter Evolution 2021

See `Twitter_NFT_Evolution_2021.pdf` for the full report.

# Usage

## Install dependencies
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 -m nltk.downloader stopwords
```
1. Create a Python virtual environment with `python3 -m venv env`
2. Activate the venv with `source env/bin/activate` (it can be deactivated by running `deactivate` or by exiting the terminal)
3. Install Python dependencies with `pip3 install -r requirements.txt`
4. Download nltk stopwords with `python3 -m nltk.downloader stopwords`

## Setup
```
unzip tweets.zip -d src/datain/clean
cd src/scripts/
python3 clean_corpus.py
cd ../..
```
1. Extract `tweets.zip`, and move the file to `datain/clean/`
    - **INPUT:** `tweets.zip` contains a [Twitter v2 Tweet object](https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet) in a **.jsonl** file.
2. In `src/scripts/`, run `python3 clean_corpus.py`. This will clean the data.

## Run BTM R code
This is for BTM topic modelling. Currently, it assumes that the number of topics is 11 (as this was determined through using the Elbow Method on the LogLik values on the dataset used).

The needed output based on the `tweets.jsonl`'s cleaned data has been run through the `BTM_implementation.Rmd` code to output `BTM_topics/data/11_model_scores.csv`. If another dataset or number of topics should be used, then the code will need to be run again.

The BTM code is a bit manual currently, but is well documented if you want to automate it a bit into a script.
* Install dependencies according to imports.
* Running LogLik (which determines how optimal the amount of topics is) is set to `False` by default (set to `True` if want to run it - this takes extra time). 
    - Need to manually take note of the LogLik values (csv with the number of topics and LogLik value is used as input in `python3 elbow_method.py`)

1. Run the `BTM_implementation.Rmd` notebook in `src/BTM_topics`.
2. You can take note of the LogLik values and then run `python3 elbow_method.py` in `src/scripts` if you want to find the most optimal number of topics for your dataset. 

## Run all Python Scripts
```
cd src/scripts
python3 run_scripts.py
```

Runs all Python scripts automatically and in order.
* Precondition: Need BTM R script `data/<numTopics>_model_scores.csv` output (ie need to run BTM R code first).
    - `<numTopics` is based on the number of topics the BTM code was run with. Default is 11.
* Does not run the Elbow Method.

## Prepping data for frequency/sentiment analysis for all topics
This is necessary to run in the Python interpreter if you wish to run any sentiment or frequency code on any given topic. By default, `run_scripts.py` only includes topic 11.
```py
import single_topic_analysis as sta
for i in range(11):
    sta.run(topic_position=i)
```

### Running topic frequency on multiple topics
After running the above, you can get each topic's frequency as well as a merged frequency graph by running the code below in the Python interpreter. By default, this will merge a graph with topics 1, 5, 6, and 7. To change this, see the `plot_frequency_merge_time()` function in `src/scripts/frequency.py`.

```py
import frequency as f
for i in range(1, 12):
    f.run(selected_topic=i) # frequency graph for an individual topic.
f.plot_frequency_merge_time() # merged frequency graph for topics 1, 5, 6, and 7.
```

Similarly, the data can also be used to plot merged sentiment graphs (see `src/notebooks/merge-sentiment-graphs.ipynb`).

## Individual scripts
Individual Python scripts can be run by running:

```
python3 <file_name>.py
```

or by importing the modules in a python interpreter in `src/scripts`:

```py
import frequency as f
f.run() # runs frequency code with default parameters
f.run(overall=True) # runs frequency code on overall data
f.run(trendline=False) # without trend line
f.run(selected_topic=3, trendline=False) # for topic 3 and without trend line
```

# Output
Final output of the Python script is generated in `src/dataout/`
* Tweet frequency over time
* Sentiment over time
* Sentiment over time merged graphs
    - Merged graph output for sentiment can be generated through running the `merge-sentiment-graphs.ipynb` notebook in `src/notebooks/`
* Topic distribution

BTM topic modelling
* BTM output can be found in the `BTM_topics/`
