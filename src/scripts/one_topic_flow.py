# Change in biggest topic overall's sentiment over time (per segment)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # plot sentiment over time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # vader sentiment analysis

from collections import Counter # count number of tweets

NUM_SEGMENTS = 35 # TODO: change this based on data

# Input/output files
# TODO: change file names based on need
INPUT_FILE = "../BTM_topics/dataout/13_model_scores.csv" # 13 topics (TODO: need to change to 11 for most optimal)
TOPIC_DISTRIBUTION_OUT = "../dataout/topic_modelling/topic_distribution_maxtopic.jpeg"
TOPIC_ID_OUT = "../datain/sentiment/maxtopic-ids.csv"
CLEANED_SENTIMENT_DATA = "../datain/sentiment/cleaned_tweets_for_sentiment.csv"
LARGEST_TOPIC_DATA_OUT = "../datain/topic_modelling/cleaned_tweets_largest_topic.csv"
TOPIC_ROUNDED_POLARITY_OUT = "../dataout/sentiment/rounded_largest_topic_sentiment.jpeg"
SENTIMENT_OVER_TIME_PER_SEGMENT_OUT = "../dataout/sentiment/sentiment_per_segment_largest_topic.jpeg"

def run():
    '''
    Run functions
    '''
    print("Largest topic flow...")
    df = topic_modelling()
    avg_sentiment = sentiment_analysis(df)

    print("Average sentiment for this topic is:", avg_sentiment)

def topic_modelling():
    '''
    Run topicmodelling related functions.
    '''
    df = load_data()
    df = match_topic_with_tweet(df)
    max_topic = get_largest_topic(df)
    plot_topic_distribution(df)
    export_topic_ids(df, max_topic)

    return df

def sentiment_analysis(df):
    '''
    Run sentiment analysis related functions.
    '''
    # sentiment analysis
    df = sentiment_get_matching_topic_data(df)
    df = clean_topic_sentiment_df(df)
    df = sentiment_polarity_score(df)
    # segments
    df, sub_dfs = split_data_segments(df)
    avg_sentiment = sentiment_per_segment(df, sub_dfs)

    return avg_sentiment

# ******************************************************************************************
# *** Topic modelling
# ******************************************************************************************

def load_data():
    '''
    Get data.

    @return df loaded data
    '''
    df = pd.read_csv(INPUT_FILE)

    # change index to id
    df = df.rename({'Unnamed: 0': 'id'}, axis=1) # rename column
    df['id'] = df['id'].astype('int64')
    df.set_index("id", inplace = True)

    # rename column headers to integer representations
    for i in range(1, len(df.columns) + 1):
        colname = "V" + str(i)
        df = df.rename({colname: i}, axis=1)
    
    return df

def match_topic_with_tweet(df):
    '''
    Get the topic that a tweet is most likely part of based on the 
    probablity that they're in the topic.

    @param df loaded df
    @return df df with a column indicating their most probable topic
    '''
    maxtopic = df
    # get the topic with the max probability value for each row
    maxtopic = maxtopic.idxmax(axis=1)
    # convert all topics from string ('15') to int (15). This prerpares it for grouping by topic
    maxtopic = maxtopic.astype(int)

    # add maxtopic as a new column
    df.insert(0, "maxtopic", maxtopic)

    # sort by maxtopic
    df = df.sort_values('maxtopic')

    return df

def get_largest_topic(df):
    '''
    Topic with the most tweets having the highest probability of being in it.

    @param df
    @return max_topic the largest topic number
    '''
    # count the number of tweets per topic (and sort in descending order)
    topic_counts = df['maxtopic'].value_counts()

    # get max topic
    max_topic = topic_counts.index[0] # TODO: change this index if want different topic
    return max_topic

def plot_topic_distribution(df):
    '''
    Count the number of occurences of each topic and plot

    @param df
    '''
    # count the number of tweets per topic using Counter
    topic2occurrences = Counter(df['maxtopic'])
    ys = []
    labels = []
    for topic, occurrences in topic2occurrences.items():
        labels.append(topic)
        ys.append(occurrences)

    plt.pie(ys, labels=labels)
    plt.title('Ratio of tweets per topic')
    # save graph
    plt.savefig(TOPIC_DISTRIBUTION_OUT)
    plt.close()

def export_topic_ids(df, max_topic):
    '''
    Get topic tweet ids and export them to a csv file.

    @param df
    @param max_topic
    '''
    max_topic_df = df.loc[df['maxtopic'] == max_topic]

    # export selected columns to csv
    selected_columns = []
    max_topic_df.to_csv(TOPIC_ID_OUT, columns = selected_columns)


# ******************************************************************************************
# *** Sentiment analysis
# ******************************************************************************************

def sentiment_get_matching_topic_data(df):
    '''
    Get the subset of the topic modelling data from the cleaned sentiment data 
    (use the topic IDs to get the sentiment data matching those ids)

    @param df
    @return largest_topic_sentiment_df
    '''
    # load cleaned tweet corpus data
    cleaned_sentiment_df = pd.read_csv(CLEANED_SENTIMENT_DATA)
    cleaned_sentiment_df = cleaned_sentiment_df.drop("Unnamed: 0", axis=1)

    # load topic ids
    max_topic_ids = pd.read_csv(TOPIC_ID_OUT)

    # subset sentiment data with topic ids
    largest_topic_sentiment_df = max_topic_ids.merge(cleaned_sentiment_df, on='id', how='left')

    # export largest topic sentiment to csv
    largest_topic_sentiment_df.to_csv(LARGEST_TOPIC_DATA_OUT)

    return largest_topic_sentiment_df

def clean_topic_sentiment_df(df):
    '''
    Clean topic subset data.
    TODO: this is the same function as in sentiment_segment.py

    @param df
    @return df dataframe with removed fields based on null values & added date/time values
    '''
    # remove all null created_at values from dataframe
    df = df.drop(df[df['created_at'].isnull()].index)
    df = df.drop(df[df['cleaned_tweet'].isnull()].index)
    # ensure that all values in created_at has 2021 (and not random strings)
    df = df[df['created_at'].str.contains("2021")]

    # split created_at into date and time columns
    #https://intellipaat.com/community/13909/python-how-can-i-split-a-column-with-both-date-and-time-e-g-2019-07-02-00-12-32-utc-into-two-separate-columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['time'] = df['created_at'].dt.time

    # make sure that all cleaned tweets are of type string
    df["cleaned_tweet"].astype(str)
    return df

def sentiment_polarity_score(df):
    '''
    Calculate sentiment vader polarity score
    '''
    analyzer = SentimentIntensityAnalyzer()

    # add polarity scores to df
    # https://github.com/sidneykung/twitter_hate_speech_detection/blob/master/preprocessing/VADER_sentiment.ipynb
    pol = lambda x: analyzer.polarity_scores(x)
    df['polarity'] = df["cleaned_tweet"].apply(pol)

    # split polarity scores into separate columns
    df = pd.concat([df.drop(['polarity'], axis=1), df['polarity'].apply(pd.Series)], axis=1)

    # get rounded polarity score
    round_pol = lambda x: calc_polarity(x, 0.05)
    # round polarity up/down
    df['rounded_polarity'] = df['compound'].apply(round_pol)

    # get amount of rounded negative, neutral, and positive polarity
    num_rounded_sentiments = df.groupby('rounded_polarity').count()
    plot_rounded_polarity(num_rounded_sentiments)

    return df

def calc_polarity(x, bound):
    '''
    Round polarity up/down based on bound.
    '''
    if x < -bound:
        return -1
    elif x > bound:
        return 1
    else:
        return 0

def plot_rounded_polarity(num_rounded_sentiments):
    '''
    Plot rounded polariry
    '''
    # plot rounded negative, neutral, and positive sentiment amounts
    plt.bar(num_rounded_sentiments.index, num_rounded_sentiments["compound"])
    plt.title('Rounded Sentiment for Largest topic')
    plt.xlabel('Polarity')
    plt.ylabel('Count')
    plt.savefig(TOPIC_ROUNDED_POLARITY_OUT)
    plt.close()

def split_data_segments(df):
    # sort dataframe by date
    df = df.sort_values(by=['date', 'time'])
    # list of dfs
    sub_dfs = list(split(df, NUM_SEGMENTS))
    return df, sub_dfs


def split(a, n):
    '''
    Split df (a) into n groups of equal length (returns list of sub dataframes)
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    '''
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def sentiment_per_segment(df, sub_dfs):
    '''
    Get average sentiment & plot sentiment over time
    '''
    compounds = []
    mns, mxs = [], []
    dates = []
    for sub_df in sub_dfs:
        compounds.append(sub_df.compound.mean())
        mxs.append(sub_df.index.max())
        mns.append(sub_df.index.min())
        dates.append(sub_df.date.iloc[0])

    compound_df = pd.DataFrame(dict(
        mn=mns,
        mx=mxs,
        compouned=compounds,
        date=dates,
    ))

    # dates = df.groupby('date').count()
    plot_sentiment_over_time(compound_df)

    # average overall sentiment
    avg_sentiment = df['compound'].mean()
    return avg_sentiment

def plot_sentiment_over_time(compound_df):
    '''
    @param compound_df
    '''
    fig, ax = plt.subplots()
    ax.plot(compound_df.date, 'compouned', data=compound_df)

    # Major ticks every month.
    fmt_month = mdates.MonthLocator()

    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    #plot
    plt.title('Sentiment per segment for largest topic (35 segments of ~3k)')
    plt.xlabel('Date')
    plt.ylabel('Vader Sentiment score')
    # save graph
    plt.savefig(SENTIMENT_OVER_TIME_PER_SEGMENT_OUT)
    plt.close()