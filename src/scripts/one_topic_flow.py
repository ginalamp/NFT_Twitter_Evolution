# Change in biggest topic overall's sentiment over time (per segment)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # plot sentiment over time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # vader sentiment analysis

from collections import Counter # count number of tweets

import sentiment_segments # sentiment analysis functions

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
    df = sentiment_segments.clean_sentiment_data(df)
    df = sentiment_segments.sentiment_polarity_score(df)
    # segments
    df, sub_dfs = sentiment_segments.split_data_segments(df)
    avg_sentiment = sentiment_segments.sentiment_per_segment(df, sub_dfs)


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