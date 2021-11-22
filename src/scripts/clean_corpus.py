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
import os # create directories

# !pip3 install nltk
# nltk.download() # run this the first time you run nltk in a python interpreter and download "stopwords"
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
TWEET_CORPUS_DATA_IN = "../datain/clean/tweets.jsonl"
FREQUENCY_DATA_OUT = "../datain/topic_modelling/cleaned_tweets_largest_community.csv"
BTM_DATA_OUT = "../datain/topic_modelling/cleaned_tweets_largest_community_btm.csv"
SENTIMENT_DATA_OUT = "../datain/sentiment/cleaned_tweets_for_sentiment.csv"

# file paths for sample data
# TWEET_CORPUS_DATA_IN = "../datain/clean/sample100k.jsonl"
# FREQUENCY_DATA_OUT = "../datain/topic_modelling/cleaned_tweets_sample.csv"
# BTM_DATA_OUT = "../datain/topic_modelling/cleaned_tweets_btm_sample.csv"
# SENTIMENT_DATA_OUT = "../datain/sentiment/cleaned_tweets_sample.csv"


def run():
    '''
        Clean corpus for sentiment and topic modelling code.
    '''
    print("Cleaning corpus...")
    create_directories()
    df = load_data()

    # cleaning for sentiment analysis (keep stop words)
    print("\tSentiment analysis cleaning...")
    remove_stop = False
    df['cleaned_tweet'] = df['corpus'].progress_apply(clean_tweet, remove_stop=remove_stop)

    print("\tWriting sentiment cleaned data to csv...")
    selected_columns = ["created_at", "id", "cleaned_tweet"] # output created_at, id, and cleaned_tweets to csv
    df.to_csv(SENTIMENT_DATA_OUT, columns = selected_columns)

    # cleaning for topic modelling (remove stop words)
    print("\tTopic modelling cleaning...")
    remove_stop = True
    df['cleaned_tweet'] = df['corpus'].progress_apply(clean_tweet, remove_stop=remove_stop)

    print("\tWriting topic modelling cleaned data to csv...")
    df.to_csv(FREQUENCY_DATA_OUT, columns = selected_columns) # frequency data needs dates
    selected_columns = ["id", "cleaned_tweet"] # BTM algorithm R script file format
    df.to_csv(BTM_DATA_OUT, columns = selected_columns, index=None)

    print("Finished cleaning corpus. The next steps will start in a few moments...")

def create_directories():
    '''
        Create input/output directories if they don't exist.
    '''
    if not os.path.exists('../datain/clean'):
        os.makedirs('../datain/clean')
    if not os.path.exists('../datain/sentiment'):
        os.makedirs('../datain/sentiment')
    if not os.path.exists('../datain/topic_modelling'):
        os.makedirs('../datain/topic_modelling')

def load_data():
    '''
        Import corpus data in json format.
        Filter to have only english tweets and remove retweets.

        Returns:
            imported english, non-retweeted data
    '''
    # import the data
    filename = TWEET_CORPUS_DATA_IN
    print("\tLoading json data...")
    print("\t\tThis can take a while (about ~10 minutes on current largest community data)")
    print("\t\tGo make yourself a cup of hot thing ;)")
    data = pd.read_json(filename, lines=True)

    # clean data: remove retweets and select only english tweets
    print("\tRemoving retweets and non-english tweets...")
    data = data[~data["text"].progress_apply(lambda x: x.startswith("RT"))]
    data = data[data["lang"].progress_apply(lambda x: x == "en")]
    data = data.rename(columns={'text': 'corpus'})
    print()

    return data


def clean_tweet(tweet, remove_stop):
    '''
        Cleans tweet by from mentions, hashtags, links, crypto wallet addresses, html entities.
            If remove_stop = True, then it cleans for topic modelling, where it removes the stop words and 
            ensures that the tweet is only letters.
            Otherwise it cleans for sentiment analysis, where it adds spaces between emojis.

        Args:
            tweet: a single tweet (String)
            remove_stop: True if stop words should be removed and False if they should not be removed.
        Returns:
            tweet: cleaned tweet (String)
    '''
    tweet = str.lower(tweet)
    tweet = ' '.join(re.sub("(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)", " ", tweet).split()) # remove mentions and hashtags
    tweet = re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub("0x([\da-z\.-]+)", "", tweet, flags=re.MULTILINE) # remove crypto wallet addresses
    tweet = re.sub('\&\w+', "", tweet) # remove html entities (example &amp)

    if remove_stop:
        # topic modelling cleaning
        tweet = re.sub('[^a-zA-Z# ]+', ' ', tweet) # make sure tweet is only letters
        tweet = remove_stopwords(tweet)
    else:
        # sentiment analysis cleaning
        tweet = add_space_between_emojis(tweet)
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

def add_space_between_emojis(tweet):
    '''
        Add spaces between emojis for the given tweet.

        Args:
            tweet: a single (cleaned) tweet (String)
        Returns:
            tweet with spaces between the emojis (String)
    '''
    # source: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
    EMOJI_PATTERN = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])"
    )
    tweet = re.sub(EMOJI_PATTERN, r' \1 ', tweet)
    return tweet

if __name__ == "__main__":
    run()
