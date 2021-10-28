# Outputs overall sentiment (with rounded polarity) and sentiment over time (frequency bins).
# Also computes overall average sentiment.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # plot sentiment over time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NUM_SEGMENTS = 40 # TODO: update based on decisions.

# Input/output files
DATA_IN = "../datain/sentiment/cleaned_tweets_for_sentiment.csv"
ROUNDED_POLARITY_OUT = "../dataout/sentiment/rounded_overall_sentiment.jpeg"
SENTIMENT_OVER_TIME_PER_SEGMENT_OUT = '../dataout/sentiment/sentiment_per_segment.jpeg'

def run():
    print("running overall sentiment analysis segments...")
    # load cleaned tweet corpus data
    df = pd.read_csv(DATA_IN)
    df = df.drop("Unnamed: 0", axis=1)

    df = clean_sentiment_data(df)

    df = sentiment_polarity_score(df)
    # segments
    df, sub_dfs = split_data_segments(df)
    avg_sentiment = sentiment_per_segment(df, sub_dfs)
    print("Average sentiment for overall is:", avg_sentiment)


def clean_sentiment_data(df):
    '''
    Load & clean data

    @return df cleaned df
    '''
    # remove all null created_at values from dataframe
    df = df.drop(df[df['created_at'].isnull()].index)
    df = df.drop(df[df['cleaned_tweet'].isnull()].index)
    # ensure that all values in created_at has 2021 (and not random strings)
    df = df[df['created_at'].str.contains("2021")]

    # split created_at into date and time columns
    # https://intellipaat.com/community/13909/python-how-can-i-split-a-column-with-both-date-and-time-e-g-2019-07-02-00-12-32-utc-into-two-separate-columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['time'] = df['created_at'].dt.time

    return df


def sentiment_polarity_score(df):
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
    plt.savefig(ROUNDED_POLARITY_OUT)
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