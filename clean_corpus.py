'''
Outputs corpus, cleaned tweets with their createdAt timestamps
into a csv.

Currently does not stem or remove stop words (code is written
for it - the user just needs to set the appropriate booleans
to True at the top.
'''

# !pip3 install nltk
import pandas as pd
import re
import html
import nltk
# nltk.download() # run this the first time you run nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer

# stop words and stemming
remove_stop = True # set this to True if want to only remove stop words
stem_remove_stop = False # set this to True if want to stem AND remove stop words

# add stop words
stop_words = stopwords.words('english')
stop_words.append('rt')
stop_words.append('nft')

def run():
    '''
    Main running code that executes all cleaning corpus functions in the
    correct order for the pipeline.
    '''
    print("Cleaning corpus...")
    df = load_data()


    # clean data text line by line
    cleaned_text = []
    for index in df.index:
        text = df["corpus"][index]
        cleaned_text.append(clean_tweet(text))
    cleaned_df = pd.DataFrame(cleaned_text[0:], columns=["cleaned_tweet"])
    df['cleaned_tweet'] = cleaned_df['cleaned_tweet']

    # output data text, cleaned_tweets, and createdAt to csv
    selected_columns = ["created_at", "corpus", "cleaned_tweet"]
    df.to_csv('datain/topic_modelling/cleaned_tweets.csv', columns = selected_columns)

    print("Finished cleaning corpus...")

def load_data():
    '''
    Import corpus data in json format.
    Filter to have only english tweets and remove retweets.

    @return imported english, non-retweeted data
    '''
    #import the data
    file_path = "datain/clean/sample100k.jsonl"
    data = pd.read_json(file_path, lines=True)

    # get first 100k lines of a larger dataset
    # data = data.iloc[0:100000] # select first 100k entries of input file

    # clean data: remove retweets and select only english tweets
    data = data[~data["text"].apply(lambda x: x.startswith("RT"))]
    data = data[data["lang"].apply(lambda x: x == "en")]
    data = data.rename(columns={'text': 'corpus'})

    return data


def clean_tweet(tweet):
    '''
    Cleans tweet from hashtags, mentions, special characters, html entities, numbers,
    links, and (optionally) stop words & stemming. Converts text to lower case.

    @param tweet - a single tweet (String)
    @return cleaned tweet (String)
    '''
    tweet = str.lower(tweet)
    tweet = ' '.join(re.sub("(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)", " ", tweet).split()) # remove mentions and hashtags
    tweet = re.sub("(http\S+|http)", "", tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub('\&\w+', "", tweet) # remove html entities (example &amp)
    tweet = re.sub('[^a-zA-Z# ]+', ' ', tweet) # make sure tweet is only letters

    # TODO - decide whether want to stem and remove stop words (default set to False)
    if remove_stop:
        tweet = remove_stopwords(tweet)
    elif stem_remove_stop:
        tweet = stem_and_remove_stopwords()
    else:
        # convert cleaned tweet to list (each tweet being one element in the list)
        tweet = ' '.join([word for word in tweet.split()])

    return tweet


def remove_stopwords(tweet):
    '''
    Remove stop words from the given tweet.

    @param tweet - a single (cleaned) tweet (String)
    @return tweet without stopwords (String)
    '''
    return' '.join([word for word in tweet.split() if word not in stop_words])

def stem_and_remove_stopwords(tweet):
    '''
    Remove stop words from the given tweet and stem the words in the tweet.

    @param tweet - a single (cleaned) tweet (String)
    @return stemmed tweet without stopwords (String)
    '''
    # TODO: perhaps use Vader's stemming function instead of PorterStemmer
    return ' '.join([PorterStemmer().stem(word=word) for word in tweet.split() if word not in stop_words])

if __name__ == "__main__":
    run()
