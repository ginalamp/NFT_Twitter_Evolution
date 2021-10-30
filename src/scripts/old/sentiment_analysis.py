'''
Sentiment analysis per topic.

TODO: This needs updating, since this was based on our old BTM script (not the new one)

Run the code if you want everything to be done automatically
all at once.
It will output the graphs into `dataout/sentiment/*`
'''
# dependencies - install first time using this notebook
# !pip3 install vaderSentiment
# !pip3 install --upgrade pip
# !pip3 install xlrd
# !pip3 install openpyxl

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# progress bar
from tqdm import tqdm



# change to False to run for a specific topic
run_all = True
numTopics = 20 # Need to set this according to the number of topics there are
# change to topic that you wish to run code for (if run_all -> False)
atopic = 6



# file paths
SENTIMENT_INPUT_FILE = "../datain/sentiment/grouped-by-topic_with_date.csv"
# SENTIMENT_INPUT_FILE = "datain/sentiment/largest_community_grouped-by-topic_with_date.csv"

def run():
    '''
    Main running code that executes all sentiment analysis functions in the
    correct order for the pipeline.
    '''
    print("Running sentiment analysis...")
    df = df_setup()
    if run_all:
        # run for all topics
        pbar = tqdm(total=numTopics)
        for topic in range(numTopics):
            sentiment_analysis_per_topic(df, topic)
            pbar.update(1)
        pbar.close()
    else:
        # run for one topic
        sentiment_analysis_per_topic(df, atopic)
    print("Finished running sentiment analysis...\nFind graphed results in dataout/sentiment/")


def df_setup():
    '''
    Set up DataFrame by reading in data, renaming columns, and adding date columns.

    @return DataFrame
    '''
    # read in data produced by topic_modelling.ipynb
    df = pd.read_csv(SENTIMENT_INPUT_FILE)
    df = df[['Unnamed: 0', 'created_at', 'maxtopic', 'corpus', "cleaned_tweet"]].copy()
    df = df.rename({'Unnamed: 0': 'tweet_index'}, axis=1)

    # remove any null created_at values from dataframe
    df = df.drop(df[df['created_at'].isnull()].index)
    # ensure that all values in created_at has 2021 (and not random strings)
    df = df[df['created_at'].str.contains("2021")]

    # split created_at into date and time columns
    #https://intellipaat.com/community/13909/python-how-can-i-split-a-column-with-both-date-and-time-e-g-2019-07-02-00-12-32-utc-into-two-separate-columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['time'] = df['created_at'].dt.time

    return df

def sentiment_analysis_per_topic(df, topic):
    '''
    Run sentiment analysis per topic.

    @param df - DataFrame on which sentiment analysis should be run on the
    cleaned_tweet according to their dates.
    @param topic - topic number on which sentiment analysis should be run.
    '''
    # filter df to only have data for the given topic
    df = df[df["maxtopic"] == topic]

    # get polarity scores per tweet
    analyzer = SentimentIntensityAnalyzer()
    for tweet in df["corpus"]:
        vs = analyzer.polarity_scores(tweet)

    #https://github.com/sidneykung/twitter_hate_speech_detection/blob/master/preprocessing/VADER_sentiment.ipynb
    pol = lambda x: analyzer.polarity_scores(x)
    df['Polarity'] = df['corpus'].apply(pol)

    df = pd.concat([df.drop(['Polarity'], axis=1), df['Polarity'].apply(pd.Series)], axis=1)
    # plot first sentiment
    plot_sentiment(df, topic)

    # plot 2nd sentiment
    round_pol = lambda x: classify_polarity(x)
    df['Polarity'] = df['compound'].apply(round_pol)
    plot_sentiment2(df, topic)

    # plot 3rd sentiment
    sentiments = df.groupby('Polarity').count()
    plot_sentiment3(sentiments, topic)

    # group tweets by date
    dates = df.groupby('date').count()
    dates['day'] = ''
    for i in range(0, len(dates.index)):
        dates['day'][i] = dates.index[i].day
    # plot tweet frequency
    plot_frequency_time(dates, topic)

    # get average tweet sentiment for a time period
    sumsentiment = 0
    count = 0
    for i in range(len(df.index)):
        if df['date'].iloc[i] == df['date'].iloc[0]:
            sumsentiment += df['compound'].iloc[i]
            count += 1
    avg = sumsentiment / count
    #calculate average sentiment per day
    #https://stackoverflow.com/questions/67899247/compute-column-average-based-on-conditions-pandas
    df_avg = df.groupby(['date'], as_index=False)['compound'].mean()

    #add day from date as separate column
    df_avg['day'] = ''
    for i in range(0, len(df_avg.index)):
        df_avg['day'][i] = df_avg['date'][i].day

    #plot average sentiment over time
    plot_sentiment_average_time(df_avg, topic)

def classify_polarity(polarity_score):
    '''
    Classify the polarity (postive, neutral, negative) according to the general
    polarity score.

    @param polarity_score - general polarity of a corpus element.
    '''
    if polarity_score < -0.05:
        return -1
    elif polarity_score > 0.05:
        return 1
    else:
        return 0

def plot_sentiment(df, topic):
    plt.plot(df.index, df['compound'].sort_values())
    plt.title('Topic {}: Sentiment Analysis using Vader'.format(topic))
    plt.ylabel('Polarity')
    plt.xlabel('Tweets')
    plt.savefig('../dataout/sentiment/Topic{}_sentiment1.jpeg'.format(topic))
    plt.close()

def plot_sentiment2(df, topic):
    plt.plot(df.index, df['Polarity'].sort_values())
    plt.title('Topic {}: Sentiment Analysis using Vader'.format(topic))
    plt.ylabel('Polarity')
    plt.xlabel('Tweets')
    plt.savefig('../dataout/sentiment/Topic{}_sentiment2.jpeg'.format(topic))
    plt.close()

def plot_sentiment3(sentiments, topic):
    plt.bar(sentiments.index, sentiments["compound"])
    plt.title('Topic {}: Sentiment Polarity Difference'.format(topic))
    plt.xlabel('Polarity')
    plt.ylabel('Count')
    plt.savefig('../dataout/sentiment/Topic{}_sentiment3.jpeg'.format(topic))
    plt.close()

def plot_frequency_time(dates, topic):
    #plot tweet frequency over time
    plt.plot(dates['day'], dates['cleaned_tweet'])
    plt.title('Topic {}: Tweet Frequency'.format(topic))
    plt.xlabel('Day in month 2021')
    plt.ylabel('Count of Topic {} Tweets'.format(topic))
    plt.savefig('../dataout/sentiment/Topic{}_frequency.jpeg'.format(topic))
    plt.close()

def plot_sentiment_average_time(df_avg, topic):
    plt.plot(df_avg['day'], df_avg['compound'])
    plt.title('Topic {}: Sentiment Over Time'.format(topic))
    plt.ylabel('Average Polarity')
    plt.xlabel('Day in month 2021')
    plt.savefig('../dataout/sentiment/Topic{}_sentiment4_time.jpeg'.format(topic))
    plt.close()

if __name__ == "__main__":
    run()
