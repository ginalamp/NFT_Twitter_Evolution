'''
    Outputs id, cleaned tweets with their createdAt timestamps
    into a csv. Also outputs file for BTM reading in format.

    Input: need to make sure that tweets.zip is extracted, moved
    to datain/clean/ and renamed to largest_community_tweets.jsonl

    Output: available in datain/topic_modelling/
'''

import pandas as pd
import re
import string

# !pip3 install nltk
# nltk.download() # run this the first time you run nltk
from nltk.corpus import stopwords

# progress bar
from tqdm import tqdm
tqdm.pandas()

TOPIC_MODELLING = 0
SENTIMENT_ANALYSIS = 1

# add stop words
stop_words = stopwords.words('english')
alpha_lower_list = list(string.ascii_lowercase)
stop_words.extend(alpha_lower_list) # appends lowercase single letters of alphabet to stopwords.
stop_words.append('rt')
stop_words.append('nft')

# file paths
TWEET_CORPUS_DATA_IN = "../datain/clean/largest_community_tweets.jsonl"
BTM_DATA_OUT = "../datain/topic_modelling/cleaned_tweets_largest_community_btm.csv"
SENTIMENT_DATA_OUT = "../datain/sentiment/cleaned_tweets_for_sentiment.csv"

# file paths for sample data
# TWEET_CORPUS_DATA_IN = "../datain/clean/sample100k.jsonl"
# BTM_DATA_OUT = "../datain/topic_modelling/cleaned_tweets.csv"
# SENTIMENT_DATA_OUT = "../datain/sentiment/cleaned_tweets.csv"


def run():
    '''
        Main running code that executes all cleaning corpus functions in the
        correct order for the pipeline.
    '''
    print("Cleaning corpus...")
    df = load_data()

    # clean data text line by line and create column with cleaned tweets
    # cleaning for sentiment analysis (keep stop words)
    print("\tCleaning tweets for sentiment analysis")
    remove_stop = False
    df['cleaned_tweet'] = df['corpus'].progress_apply(clean_tweet, remove_stop=remove_stop)
    selected_columns = ["created_at", "id", "cleaned_tweet"] # output created_at, id, and cleaned_tweets to csv
    print("\twriting sentiment cleaned data to csv...")
    df.to_csv(SENTIMENT_DATA_OUT, columns = selected_columns)

    # cleaning for topic modelling (remove stop words)
    print("\tCleaning tweets for topic modelling")
    remove_stop = True
    df['cleaned_tweet'] = df['corpus'].progress_apply(clean_tweet, remove_stop=remove_stop)
    selected_columns = ["id", "cleaned_tweet"] # BTM algorithm R script file format
    print("\twriting topic modelling cleaned data to csv...")
    df.to_csv(BTM_DATA_OUT, columns = selected_columns, index=None)

    print("Finished cleaning corpus...")

def load_data():
    '''
        Import corpus data in json format.
        Filter to have only english tweets and remove retweets.

        Returns:
            imported english, non-retweeted data
    '''
    #import the data
    filename = TWEET_CORPUS_DATA_IN
    print("\tLoading json data...")
    print("\t\tThis can take a while (about ~10 minutes on current largest community data)")
    print("\t\tGo make yourself a cup of hot thing ;)")
    data = pd.read_json(filename, lines=True)

    # clean data: remove retweets and select only english tweets
    print("\tRemoving reweets and non-english tweets...")
    data = data[~data["text"].progress_apply(lambda x: x.startswith("RT"))]
    data = data[data["lang"].progress_apply(lambda x: x == "en")]
    data = data.rename(columns={'text': 'corpus'})
    print()

    return data


def clean_tweet(tweet, remove_stop):
    '''
        Cleans tweet from hashtags, mentions, special characters, html entities, numbers,
        links, and stop words. Converts text to lower case.

        Args:
            tweet: a single tweet (String)
            remove_stop: True if stopwords should be removed and False if they should not be removed.
        Returns:
            tweet: cleaned tweet (String)
    '''
    tweet = str.lower(tweet)
    tweet = ' '.join(re.sub("(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)", " ", tweet).split()) # remove mentions and hashtags
    tweet = re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub("0x([\da-z\.-]+)", "", tweet, flags=re.MULTILINE) # remove addresses/pointers
    tweet = re.sub('\&\w+', "", tweet) # remove html entities (example &amp)
    tweet = re.sub('[^a-zA-Z# ]+', ' ', tweet) # make sure tweet is only letters

    if remove_stop:
        tweet = remove_stopwords(tweet)
    else:
        # convert cleaned tweet to list (each tweet being one element in the list)
        tweet = ' '.join([word for word in tweet.split()])

    return tweet


def remove_stopwords(tweet):
    '''
        Remove stop words from the given tweet.

        Args:
            tweet: a single (cleaned) tweet (String)
        Returns:
            tweet without stopwords (String)
    '''
    return' '.join([word for word in tweet.split() if word not in stop_words])

if __name__ == "__main__":
    run()
