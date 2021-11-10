'''
    Plot tweet frequency over time for large dataset
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np

# file paths
OVERALL_DATA_IN = "../datain/topic_modelling/cleaned_tweets_largest_community.csv" # overall tweets
OVERALL_DATA_OUT = "../dataout/general/tweet_frequency_overall.pdf" # overall tweets

TOPIC_DATA_IN_PREFIX = "../datain/topic_modelling/" # topic
TOPIC_DATA_OUT_PREFIX = "../dataout/general/" # topic

# file paths for sample data
# OVERALL_DATA_IN = "../datain/topic_modelling/cleaned_tweets.csv"
# OVERALL_DATA_OUT = "../dataout/general/Total_tweet_frequency.jpeg"

def run(overall=False, selected_topic=11, trendline=True):
    '''
        Run frequency code.
        Default run for largest topic.

        Args:
            overall: boolean (true if want to analyse overall data frequency, false if not)
            selected_topic: the selected topic number
            trendline: boolean (true if a trendline should be plotted, false otherwise)
    '''
    if not overall:
        print("Calculating selected topic tweet frequency...")
        print("\tSetting topic I/O files...")
        # data_in = TOPIC_DATA_IN_PREFIX + f"tweet_sentiment_subdf_topic_{selected_topic}.csv"
        data_in = TOPIC_DATA_IN_PREFIX + f"tweet_topic_subdf_topic_{selected_topic}.csv"
        data_out = TOPIC_DATA_OUT_PREFIX + f"tweet_frequency_topic_{selected_topic}.pdf"
    else:
        print("Calculating overall tweet frequency...")
        print("\tSetting overall I/O files...")
        data_in = OVERALL_DATA_IN
        data_out = OVERALL_DATA_OUT
        overall = True

    # load tweet corpus data
    df = pd.read_csv(data_in)
    df = df.drop("Unnamed: 0", axis=1)

    # remove any null created_at values from dataframe
    df = df.drop(df[df['created_at'].isnull()].index)
    # ensure that all values in created_at has 2021 (and not random strings)
    df = df[df['created_at'].str.contains("2021")]

    # split created_at into date and time columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['time'] = df['created_at'].dt.time

    # group tweets by date and count number of entries per day
    dates = df.groupby('date').count()
    plot_frequency_time(dates, overall, selected_topic, data_out, trendline)
    print("\tGraph outputs available in dataout/")


def plot_frequency_time(dates, overall, selected_topic, data_out, trendline):
    '''
        Plot tweet frequency over time.

        Args:
            dates: df with count of number of tweets posted grouped by date
            overall: True if is for an overall analysis, False if it is for a topic's analysis.
            selected_topic: the selected topic number
            data_out: path to the file to which this function will output to.
            trendline: boolean (true if a trendline should be plotted, false otherwise)
    '''
    fig, ax = plt.subplots()
    ax.plot(dates.index, 'cleaned_tweet', data=dates)

    # add trendline
    ax2 = ax.twiny()
    z = np.polyfit(range(len(dates["cleaned_tweet"])), dates['cleaned_tweet'], 3)
    p = np.poly1d(z)
    x = range(len(dates["cleaned_tweet"]))
    if trendline:
        plt.plot(x, p(x), color='orange')

    # Major ticks every month.
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(ticker.NullLocator())
    # plot
    if overall:
        plt.title('Overall Tweet Frequency over time')
    else:
        plt.title(f'Topic {selected_topic}: Tweet Frequency over time')
        ax.set_xlabel('Date')

    plt.ylabel('Number of Tweets')
    plt.savefig(data_out)
    plt.close()

if __name__ == "__main__":
    #run(selected_topic=1, trendline=False)
    run(overall=True)
