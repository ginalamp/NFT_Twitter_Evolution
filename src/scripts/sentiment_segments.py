'''
    Outputs overall sentiment (with rounded polarity) and sentiment over time (frequency bins).
    Also computes overall average sentiment.
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # plot sentiment over time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# progress bar
from tqdm import tqdm
tqdm.pandas()

NUM_SEGMENTS = 34 # decided on 34 segments for overall data

# Input/output files for overall data
DATA_IN = "../datain/sentiment/cleaned_tweets_for_sentiment.csv"
ROUNDED_POLARITY_OUT = "../dataout/sentiment/rounded_sentiment_overall.jpeg"
SENTIMENT_OVER_TIME_PER_SEGMENT_OUT = '../dataout/sentiment/sentiment_per_segment.jpeg'

def run(overall=True):
    '''
        Runs functions apply sentiment analysis on segments over time.

        Args:
            overall: boolean (true if want to analyse overall data frequency, false if not)
    '''
    print("Running overall sentiment analysis segments...")
    # load cleaned tweet corpus data
    df = pd.read_csv(DATA_IN)
    df = df.drop("Unnamed: 0", axis=1)

    df = clean_sentiment_data(df)

    df = sentiment_polarity_score(df)
    # segments
    df, sub_dfs, num_segments = split_data_segments(df)
    num_tweets_per_segment = round(len(sub_dfs[0]) / 1000, 1)
    avg_sentiment = sentiment_per_segment(df, sub_dfs, num_segments, num_tweets_per_segment, overall)
    print("\tAverage sentiment overall is:", avg_sentiment)


def clean_sentiment_data(df):
    '''
        Load & clean data.

        Args:
            df: dataframe containing the sentiment data
        Returns:
            df: cleaned dataframe
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


def sentiment_polarity_score(df, filename=ROUNDED_POLARITY_OUT):
    '''
        Calculates the sentiment polarity score.

        Args:
            df: cleaned dataframe with tweet data
            filename: path to the file to which this function will output to.
        Returns:
            df: dataframe with Vader sentiment polarity score columns added.
    '''
    analyzer = SentimentIntensityAnalyzer()

    # add polarity scores to df
    # https://github.com/sidneykung/twitter_hate_speech_detection/blob/master/preprocessing/VADER_sentiment.ipynb
    print(f"\t\tGetting sentiment polarity scores...")
    pol = lambda x: analyzer.polarity_scores(x)
    df['polarity'] = df["cleaned_tweet"].progress_apply(pol)

    # split polarity scores into separate columns
    print(f"\t\tSplitting polarity scores into dataframe columns...")
    df = pd.concat([df.drop(['polarity'], axis=1), df['polarity'].progress_apply(pd.Series)], axis=1)

    # get rounded polarity score
    round_pol = lambda x: calc_polarity(x, 0.05)
    # round polarity up/down
    print(f"\t\tGet rounded polarity...")
    df['rounded_polarity'] = df['compound'].progress_apply(round_pol)

    # get amount of rounded negative, neutral, and positive polarity
    num_rounded_sentiments = df.groupby('rounded_polarity').count()
    plot_rounded_polarity(num_rounded_sentiments, filename)

    return df


def calc_polarity(x, bound):
    '''
        Round polarity up/down based on bound.

        Args:
            x: 
            bound:
        Returns:
            int: -1 if x is less than -bound, 1 greater than bound, or 0
    '''
    if x < -bound:
        return -1
    elif x > bound:
        return 1
    else:
        return 0

def plot_rounded_polarity(num_rounded_sentiments, filename):
    '''
        Plot rounded polariry.
        Called by sentiment_polarity_score().

        Args:
            num_rounded_sentiments: dataframe grouped by rounded polarity
            filename: path to the file to which this function will output to.
    '''
    # plot rounded negative, neutral, and positive sentiment amounts
    plt.bar(num_rounded_sentiments.index, num_rounded_sentiments["compound"])
    plt.title('Rounded Sentiment for Largest topic')
    plt.xlabel('Polarity')
    plt.ylabel('Count')
    plt.savefig(filename)
    plt.close()


def split_data_segments(df, num_segments=NUM_SEGMENTS):
    '''
        Split data into segments according to date.

        Args:
            df: dataframe with Vader sentiment polarity score columns added.
            num_segments: number of equal segments that the data needs to be split into.
        Returns:
            df: sorted df by date
            sub_dfs: a list of subdataframes of df
            num_segments: number of equal segments that the data needs to be split into.
    '''
    # sort dataframe by date
    df = df.sort_values(by=['date', 'time'])
    # list of dfs
    sub_dfs = list(split(df, num_segments))
    return df, sub_dfs, num_segments


def split(df, n):
    '''
        Split df into n groups of equal length (returns list of sub dataframes).
        https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

        Args:
            df: dataframe that should be split
            n: number of equal segments that the data needs to be split into.
        Retuns:
            sub dataframe according to df and n
    '''
    k, m = divmod(len(df), n)
    return (df[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def sentiment_per_segment(df, sub_dfs, num_segments, num_tweets_per_segment, overall, filename=SENTIMENT_OVER_TIME_PER_SEGMENT_OUT):
    '''
        Get average sentiment & plot sentiment over time.

        Args:
            df: sorted df by date
            sub_dfs: a list of subdataframes of df
            num_segments: number of equal segments that the data needs to be split into.
            num_tweets_per_segment: number of tweets per segment.
            overall: boolean (true if want to analyse overall data frequency, false if not)
            filename: path to the file to which this function will output to.
        Returns:
            avg_sentiment: the average sentiment over the entire timeperiod for the data.
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

    plot_sentiment_over_time(compound_df, num_segments, num_tweets_per_segment, overall, filename)

    # average overall sentiment
    avg_sentiment = df['compound'].mean()
    return avg_sentiment

def plot_sentiment_over_time(compound_df, num_segments, num_tweets_per_segment, overall, filename):
    '''
        Plot sentiment over time.

        Args:
            compound_df: dataframe with the compound Vader sentiment value for each segment. 
            num_segments: number of equal segments that the data needs to be split into.
            num_tweets_per_segment:
            overall: boolean (true if want to analyse overall data frequency, false if not)
            filename: path to the file to which this function will output to.
    '''
    fig, ax = plt.subplots()
    ax.plot(compound_df.date, 'compouned', data=compound_df)

    # Major ticks every month.
    fmt_month = mdates.MonthLocator()

    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # plot
    if overall:
        # plt.title('Overall Tweet Frequency over time: 1 Feb - 31 May')
        plt.title('Sentiment per segment overall ({} segments of ~{}k)'.format(num_segments, num_tweets_per_segment))
        
    else:
        # TODO: edit this such that it can be any topic
        plt.title('Sentiment per segment for largest topic ({} segments of ~{}k)'.format(num_segments, num_tweets_per_segment))
        
    # plt.title('Sentiment per segment for largest topic ({} segments of ~{}k)'.format(NUM_SEGMENTS, num_tweets_per_segment))
    # plt.title('Sentiment per segment for largest topic (35 segments of ~3k)')
    plt.xlabel('Date')
    plt.ylabel('Vader Sentiment score')
    # save graph
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    run()