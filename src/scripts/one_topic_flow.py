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

NUM_SEGMENTS = 40 # decided on 40 segments for largest topic from ../BTM_topics/dataout/11_model_scores.csv

# Input/output files
BTM_SCORES_DATA_IN = "../BTM_topics/dataout/"
BTM_DATA_IN_PREFIX = "../datain/topic_modelling/"
BTM_DATA_OUT_PREFIX = "../dataout/topic_modelling/"

SENTIMENT_DATA_IN_PREFIX = "../datain/sentiment/"
SENTIMENT_DATA_OUT_PREFIX = "../dataout/sentiment/"

def run(topic_position=0, optimal_num_topics=11):
    '''
        Run functions.

        Args:
            topic_position: integer (0 for largest topic, 1 for second largest, etc.)
            optimal_num_topics: optimal number of topics identified by the ElbowMethod (using the R BTM LogLik values)
                - default is 11 topics, since it is the most optimal from the data this was run.
    '''
    print(f"Running normalised optimal topic flow")
    print(f"\tTopic position: {topic_position}")
    df, selected_topic = topic_modelling(topic_position, optimal_num_topics)
    avg_sentiment = sentiment_analysis(df, selected_topic)

    print(f"\tAverage sentiment for topic {selected_topic} is: {avg_sentiment}")

def topic_modelling(topic_position, optimal_num_topics):
    '''
        Run topic modelling related functions.

        Args:
            topic_position: integer (0 for largest topic, 1 for second largest, etc.)
            optimal_num_topics: optimal number of topics identified by the ElbowMethod (using the R BTM LogLik values)
    '''
    df = load_data(optimal_num_topics)
    df = match_topic_with_tweet(df)
    selected_topic = get_selected_topic(df, topic_position)
    print(f"\tThe selected topic is: {selected_topic}")
    plot_topic_distribution(df)
    export_topic_ids(df, selected_topic)

    return df, selected_topic

def sentiment_analysis(df, selected_topic):
    '''
        Run sentiment analysis related functions.

        Args:
            df:
            selected_topic:
    '''
    print("\tGetting topic sentiment...")
    # sentiment analysis
    df = sentiment_get_matching_topic_data(df, selected_topic)
    print("a")
    df = sentiment_segments.clean_sentiment_data(df)
    print("b")
    filename = SENTIMENT_DATA_OUT_PREFIX + f"rounded_sentiment_topic_{selected_topic}.jpeg"
    df = sentiment_segments.sentiment_polarity_score(df, filename)
    # segments
    print("c")
    df, sub_dfs, num_segments = sentiment_segments.split_data_segments(df, NUM_SEGMENTS)
    print("d")
    num_tweets_per_segment = round(len(sub_dfs[0]) / 1000, 1)
    filename = SENTIMENT_DATA_OUT_PREFIX + f"sentiment_per_segment_topic_{selected_topic}.jpeg"
    print("e")
    avg_sentiment = sentiment_segments.sentiment_per_segment(df, sub_dfs, num_segments, num_tweets_per_segment, False, filename)
    print("f")

    return avg_sentiment

# ******************************************************************************************
# *** Topic modelling
# ******************************************************************************************

def load_data(optimal_num_topics):
    '''
        Get data.

        Args:
            optimal_num_topics: optimal number of topics identified by the ElbowMethod (using the R BTM LogLik values)
        Returns
            df: loaded data
    '''
    filename = BTM_SCORES_DATA_IN + f"{optimal_num_topics}_model_scores.csv"
    df = pd.read_csv(filename)

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

def get_selected_topic(df, topic_position):
    '''
        Topic with the most tweets having the highest probability of being in it.

        Args:
            df:
            topic_position: integer (0 for largest topic, 1 for second largest, etc.)
        Returns:
            selected_topic: the selected topic number
    '''
    # count the number of tweets per topic (and sort in descending order)
    topic_counts = df['maxtopic'].value_counts()

    # get max topic
    selected_topic = topic_counts.index[topic_position]
    return selected_topic

def plot_topic_distribution(df):
    '''
        Count the number of occurences of each topic and plot

        Args:
            df:
    '''
    filename = BTM_DATA_OUT_PREFIX + "topic_distribution_overall.jpeg"

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

def export_topic_ids(df, selected_topic):
    '''
        Get topic tweet ids and export them to a csv file.

        Args:
            df:
            selected_topic: topic number
    '''
    filename = SENTIMENT_DATA_IN_PREFIX + "maxtopic-ids.csv"
    selected_topic_df = df.loc[df['maxtopic'] == selected_topic]

    # export selected columns to csv
    selected_columns = []
    selected_topic_df.to_csv(filename, columns = selected_columns)


# ******************************************************************************************
# *** Sentiment analysis
# ******************************************************************************************

def sentiment_get_matching_topic_data(df, selected_topic):
    '''
        Get the subset of the topic modelling data from the cleaned sentiment data 
        (use the topic IDs to get the sentiment data matching those ids).

        Args:
            df:
            selected_topic:
        Returns:
            selected_topic_sentiment_df
    '''
    filename = SENTIMENT_DATA_IN_PREFIX + "cleaned_tweets_for_sentiment.csv"
    # load cleaned tweet corpus data
    cleaned_sentiment_df = pd.read_csv(filename)
    cleaned_sentiment_df = cleaned_sentiment_df.drop("Unnamed: 0", axis=1)

    # load topic ids
    filename = SENTIMENT_DATA_IN_PREFIX + "maxtopic-ids.csv"
    selected_topic_ids = pd.read_csv(filename)

    # subset sentiment data with topic ids
    selected_topic_sentiment_df = selected_topic_ids.merge(cleaned_sentiment_df, on='id', how='left')

    # export selected topic sentiment to csv
    filename = BTM_DATA_IN_PREFIX + f"cleaned_tweets_topic_{selected_topic}.csv"
    selected_topic_sentiment_df.to_csv(filename)

    return selected_topic_sentiment_df

if __name__ == "__main__":
    run()