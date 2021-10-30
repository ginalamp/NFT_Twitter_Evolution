'''
    Change in biggest topic overall's sentiment over time (per segment)
'''

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates # plot sentiment over time

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # vader sentiment analysis

from collections import Counter # count number of tweets

import sentiment_segments # sentiment analysis functions

NUM_SEGMENTS = 40 # decided on 40 segments for largest topic
TOPIC_POSITION = 0

# Input/output files
BTM_SCORES_DATA_IN = "../BTM_topics/dataout/11_model_scores.csv" # 11 topics is the most optimal
BTM_DATA_IN_PREFIX = "../datain/topic_modelling/"
BTM_DATA_OUT_PREFIX = "../dataout/topic_modelling/"

SENTIMENT_DATA_IN_PREFIX = "../datain/sentiment/"
SENTIMENT_DATA_OUT_PREFIX = "../dataout/sentiment/"

def run(topic_position=0):
    '''
        Run functions.

        Args:
            topic_position: integer (0 for largest topic, 1 for second largest, etc.)
    '''
    print("Largest topic flow...")
    TOPIC_POSITION = topic_position
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

        Args:
            df:
    '''
    # sentiment analysis
    df = sentiment_get_matching_topic_data(df)
    df = sentiment_segments.clean_sentiment_data(df)
    filename = SENTIMENT_DATA_OUT_PREFIX + "rounded_largest_topic_sentiment.jpeg"
    df = sentiment_segments.sentiment_polarity_score(df, filename)
    # segments
    df, sub_dfs = sentiment_segments.split_data_segments(df, NUM_SEGMENTS)
    num_tweets_per_segment = round(len(sub_dfs[0]) / 1000, 1)
    filename = SENTIMENT_DATA_OUT_PREFIX + "sentiment_per_segment_largest_topic.jpeg"
    avg_sentiment = sentiment_segments.sentiment_per_segment(df, sub_dfs, num_tweets_per_segment, filename)


    return avg_sentiment

# ******************************************************************************************
# *** Topic modelling
# ******************************************************************************************

def load_data():
    '''
        Get data.

        Returns
            df: loaded data
    '''
    df = pd.read_csv(BTM_SCORES_DATA_IN)

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

        Args:
            df: loaded df
        Returns:
            df: df with a column indicating their most probable topic
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

        Args:
            df:
        Returns:
            max_topic: the largest topic number
    '''
    # count the number of tweets per topic (and sort in descending order)
    topic_counts = df['maxtopic'].value_counts()

    # get max topic
    max_topic = topic_counts.index[TOPIC_POSITION] # TODO: update function & variable names to indicate that it is the specified topic position (not always largest)
    return max_topic

def plot_topic_distribution(df):
    '''
        Count the number of occurences of each topic and plot

        Args:
            df:
    '''
    filename = BTM_DATA_OUT_PREFIX + "topic_distribution_maxtopic.jpeg"

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
    plt.savefig(filename)
    plt.close()

def export_topic_ids(df, max_topic):
    '''
        Get topic tweet ids and export them to a csv file.

        Args:
            df:
        Returns:
            max_topic
    '''
    filename = SENTIMENT_DATA_IN_PREFIX + "maxtopic-ids.csv"
    max_topic_df = df.loc[df['maxtopic'] == max_topic]

    # export selected columns to csv
    selected_columns = []
    max_topic_df.to_csv(filename, columns = selected_columns)


# ******************************************************************************************
# *** Sentiment analysis
# ******************************************************************************************

def sentiment_get_matching_topic_data(df):
    '''
        Get the subset of the topic modelling data from the cleaned sentiment data 
        (use the topic IDs to get the sentiment data matching those ids).

        Args:
            df:
        Returns:
            largest_topic_sentiment_df
    '''
    filename = SENTIMENT_DATA_IN_PREFIX + "cleaned_tweets_for_sentiment.csv"
    # load cleaned tweet corpus data
    cleaned_sentiment_df = pd.read_csv(filename)
    cleaned_sentiment_df = cleaned_sentiment_df.drop("Unnamed: 0", axis=1)

    # load topic ids
    filename = SENTIMENT_DATA_IN_PREFIX + "maxtopic-ids.csv"
    max_topic_ids = pd.read_csv(filename)

    # subset sentiment data with topic ids
    largest_topic_sentiment_df = max_topic_ids.merge(cleaned_sentiment_df, on='id', how='left')

    # export largest topic sentiment to csv
    filename = BTM_DATA_IN_PREFIX + "cleaned_tweets_largest_topic.csv"
    largest_topic_sentiment_df.to_csv(filename)

    return largest_topic_sentiment_df

if __name__ == "__main__":
    run()