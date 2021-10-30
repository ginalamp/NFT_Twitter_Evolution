'''
    Outputs id, cleaned tweets with their createdAt timestamps
    into a csv. Also outputs file for BTM reading in format.

    Input: need to make sure that tweets.zip is extracted, moved
    to datain/clean/ and renamed to largest_community_tweets.jsonl

    Output: available in datain/sentiment/
'''

# TODO: merge this with clean_corpus_for_btm.py (only difference is not stopword removal)
import pandas as pd
import re

# progress bar
from tqdm import tqdm
tqdm.pandas()

# file paths
DATA_IN = "../datain/clean/largest_community_tweets.jsonl"
DATA_OUT = "../datain/sentiment/cleaned_tweets_for_sentiment.csv"

# file paths for sample data
# TWEET_CORPUS_INPUT_FILE = "datain/clean/sample100k.jsonl"
# CLEANED_TWEETS_OUTPUT_FILE = "datain/sentiment_analysis/cleaned_tweets.csv"

def run():
    '''
        Main running code that executes all cleaning corpus functions in the
        correct order for the pipeline.
    '''
    print("Cleaning corpus for sentiment analysis...")
    df = load_data()

    # clean data text line by line and create column with cleaned tweets
    print("\tCleaning tweets")
    df['cleaned_tweet'] = df['corpus'].progress_apply(clean_tweet)

    # output id, cleaned_tweets, and createdAt to csv
    selected_columns = ["created_at", "id", "cleaned_tweet"]
    df.to_csv(DATA_OUT, columns = selected_columns)

    print("\tFinished cleaning corpus...")

def load_data():
    '''
        Import corpus data in json format.
        Filter to have only english tweets and remove retweets.

        Returns:
            imported english, non-retweeted data
    '''
    #import the data
    filename = DATA_IN
    print("\tLoading json data")
    data = pd.read_json(filename, lines=True)

    # clean data: remove retweets and select only english tweets
    print("\tRemoving reweets and non-english tweets")
    data = data[~data["text"].progress_apply(lambda x: x.startswith("RT"))]
    data = data[data["lang"].progress_apply(lambda x: x == "en")]
    data = data.rename(columns={'text': 'corpus'})

    return data

def clean_tweet(tweet):
    '''
        Cleans tweet from hashtags, mentions, special characters, html entities, numbers,
        links (not stopwords). Converts text to lower case.

        Args:
            tweet: a single tweet (String)
        Returns:
            tweet: cleaned tweet (String)
    '''
    tweet = str.lower(tweet)
    tweet = ' '.join(re.sub("(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)", " ", tweet).split()) # remove mentions and hashtags
    tweet = re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub("0x([\da-z\.-]+)", "", tweet, flags=re.MULTILINE) # remove addresses/pointers
    tweet = re.sub('\&\w+', "", tweet) # remove html entities (example &amp)
    tweet = re.sub('[^a-zA-Z#,.?!\-\'();: ]+', ' ', tweet) # make sure tweet is only letters and punctuation

    # convert cleaned tweet to list (each tweet being one element in the list)
    tweet = ' '.join([word for word in tweet.split()])

    return tweet


if __name__ == "__main__":
    run()
