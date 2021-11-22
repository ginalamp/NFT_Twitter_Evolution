'''
    Change in biggest topic overall's sentiment over time (per segment)
'''

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter # count number of tweets

import sentiment_segments # sentiment analysis functions

NUM_SEGMENTS = 40 # decided on 40 segments for largest topic from ../BTM_topics/dataout/11_model_scores.csv

# Input/output files
BTM_SCORES_DATA_IN = "../BTM_topics/dataout/"
BTM_DATA_IN_PREFIX = "../datain/topic_modelling/"
BTM_DATA_OUT_PREFIX = "../dataout/topic_modelling/"

SENTIMENT_DATA_IN_PREFIX = "../datain/sentiment/"
SENTIMENT_DATA_OUT_PREFIX = "../dataout/sentiment/"

# no sample file input/output since there are too many and needs editing of BTM R file. Deal with it.

def run(topic_position=0, optimal_num_topics=11):
    '''
        Run functions.

        Args:
            topic_position: integer (0 for largest topic, 1 for second largest, etc.)
            optimal_num_topics: optimal number of topics identified by the ElbowMethod (using the R BTM LogLik values)
                - default is 11 topics, since it is the most optimal from the data this was run.
        Returns:
            selected_topic: the selected topic number
    '''
    print(f"Applying topic modelling & sentiment analysis on a single topic...")
    print(f"\tTopic position: {topic_position}")
    df, selected_topic = topic_modelling(topic_position, optimal_num_topics)
    avg_sentiment = sentiment_analysis(df, selected_topic)

    print(f"\tAverage sentiment for topic {selected_topic} is: {avg_sentiment}")
    print("\tOutput available in dataout/sentiment/")
    
    return selected_topic

def topic_modelling(topic_position, optimal_num_topics=11):
    '''
        Run topic modelling related functions.

        Args:
            topic_position: integer (0 for largest topic, 1 for second largest, etc.)
            optimal_num_topics: optimal number of topics identified by the ElbowMethod (using the R BTM LogLik values)
        Returns:
            df: dataframe with topic numbers corresponding to their tweets.
            selected_topic: the selected topic number
    '''
    df = load_data(optimal_num_topics)
    df = match_topic_with_tweet(df)
    selected_topic = get_selected_topic(df, topic_position)
    print(f"\tThe selected topic is: {selected_topic}")
    plot_topic_distribution(df)
    export_topic_ids(df, selected_topic)
    cleaned_community_get_matching_topic_data(selected_topic)

    print("\tOutput available in dataout/topic_modelling/")
    return df, selected_topic

def sentiment_analysis(df, selected_topic):
    '''
        Run sentiment analysis related functions.

        Args:
            df: dataframe with topic numbers corresponding to their tweets.
            selected_topic: the topic number of the topic to be analysed.
        Returns:
            avg_sentiment: the average sentiment for the selected topic over the time period.
    '''
    print("\tGetting topic sentiment...")
    # sentiment analysis
    df = sentiment_get_matching_topic_data(selected_topic)
    df = sentiment_segments.clean_sentiment_data(df)
    # filename = SENTIMENT_DATA_OUT_PREFIX + f"rounded_sentiment_topic_{selected_topic}.jpeg"
    filename = SENTIMENT_DATA_OUT_PREFIX + f"rounded_sentiment_topic_{selected_topic}.pdf"

    df = sentiment_segments.sentiment_polarity_score(df, False, selected_topic, filename)
    # segments
    df, sub_dfs, num_segments = sentiment_segments.split_data_segments(df, NUM_SEGMENTS)
    num_tweets_per_segment = round(len(sub_dfs[0]) / 1000, 1)
    # filename = SENTIMENT_DATA_OUT_PREFIX + f"sentiment_per_segment_topic_{selected_topic}.jpeg"
    filename = SENTIMENT_DATA_OUT_PREFIX + f"sentiment_per_segment_topic_{selected_topic}.pdf"
    avg_sentiment = sentiment_segments.sentiment_per_segment(df, sub_dfs, num_segments, num_tweets_per_segment, False, selected_topic, filename)

    return avg_sentiment

# ******************************************************************************************
# *** Topic modelling
# ******************************************************************************************

def load_data(optimal_num_topics=11):
    '''
        Get data.

        Args:
            optimal_num_topics: optimal number of topics identified by the ElbowMethod (using the R BTM LogLik values)
        Returns
            df: loaded BTM scores dataframe
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
            df: loaded BTM scores dataframe
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
        Get the topic with the according to the topic_position that having the highest probability of being in that topic.
            For example:
                if topic_position is 0 it will get the topic with the most amount of tweets associated with it,
                if topic_position is 1 it will get the topic with the second most amount of tweets associated with it,
                etc.

        Args:
            df: df with a column indicating their most probable topic
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
        Plot the distribution of tweets associated with each topic.

        Args:
            df: df with a column indicating their most probable topic 
    '''
    # filename = BTM_DATA_OUT_PREFIX + "topic_distribution_overall.jpeg"
    filename = BTM_DATA_OUT_PREFIX + "topic_distribution_overall.pdf"
    

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
            df: df with a column indicating their most probable topic
            selected_topic: topic number who's IDs should be exported
    '''
    filename = SENTIMENT_DATA_IN_PREFIX + f"ids_topic_{selected_topic}.csv"
    selected_topic_df = df.loc[df['maxtopic'] == selected_topic]

    # export selected columns to csv
    selected_columns = []
    selected_topic_df.to_csv(filename, columns = selected_columns)

def cleaned_community_get_matching_topic_data(selected_topic):
    '''
        Get the subset of the topic modelling data from the cleaned topic modelling data 
        (use the topic IDs to get the overal cleaned topic data matching those ids).

        Args:
            selected_topic: the topic number of the topic to be analysed.
    '''
    filename = "../datain/topic_modelling/cleaned_tweets_largest_community.csv" # overall tweets
    # load cleaned btm tweet corpus data
    cleaned_btm_df = pd.read_csv(filename)
    cleaned_btm_df = cleaned_btm_df.drop("Unnamed: 0", axis=1)

    # load topic ids
    filename = SENTIMENT_DATA_IN_PREFIX + f"ids_topic_{selected_topic}.csv"
    selected_topic_ids = pd.read_csv(filename)

    # subset overall topic data with topic ids
    selected_topic_btm_df = selected_topic_ids.merge(cleaned_btm_df, on='id', how='left')

    # export selected topic to csv
    filename = BTM_DATA_IN_PREFIX + f"tweet_topic_subdf_topic_{selected_topic}.csv"
    selected_topic_btm_df.to_csv(filename)

# ******************************************************************************************
# *** Sentiment analysis
# ******************************************************************************************

def sentiment_get_matching_topic_data(selected_topic):
    '''
        Get the subset of the topic modelling data from the cleaned sentiment data 
        (use the topic IDs to get the sentiment data matching those ids).

        Args:
            selected_topic: the topic number of the topic to be analysed.
        Returns:
            selected_topic_sentiment_df: subset of cleaned sentiment data that matches the selected topic's tweet ids.
    '''
    filename = SENTIMENT_DATA_IN_PREFIX + "cleaned_tweets_for_sentiment.csv"
    # load cleaned tweet corpus data
    cleaned_sentiment_df = pd.read_csv(filename)
    cleaned_sentiment_df = cleaned_sentiment_df.drop("Unnamed: 0", axis=1)

    # load topic ids
    filename = SENTIMENT_DATA_IN_PREFIX + f"ids_topic_{selected_topic}.csv"
    selected_topic_ids = pd.read_csv(filename)

    # subset sentiment data with topic ids
    selected_topic_sentiment_df = selected_topic_ids.merge(cleaned_sentiment_df, on='id', how='left')

    # export selected topic sentiment to csv
    filename = BTM_DATA_IN_PREFIX + f"tweet_sentiment_subdf_topic_{selected_topic}.csv"
    selected_topic_sentiment_df.to_csv(filename)

    return selected_topic_sentiment_df

if __name__ == "__main__":
    run()