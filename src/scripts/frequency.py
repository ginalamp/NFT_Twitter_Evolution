'''
    Plot tweet frequency over time. 
    Can run for overall and for a selected topic's dataset (default topic 11).

    Outputs frequency graphs & csv's with more than NUM_TWEETS_THRESHOLD.
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

GROUP_BY = "date" # NOTE: change this to week/month if want to get the frequency increase in week/month
NUM_TWEETS_THRESHOLD = 15000 # outputs data with GROUP_BY (date) higher than this number

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
        print(f"Calculating selected topic tweet frequency for topic {selected_topic}...")
        print("\tSetting topic I/O files...")
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
    df['date'] = df['created_at'].dt.date # default grouping
    df['time'] = df['created_at'].dt.time
    df['month'] = df['created_at'].dt.month # can group by month
    df['week'] = df['created_at'].dt.isocalendar().week # can group by week

    # group tweets by date and count number of entries per day
    dates = df.groupby(GROUP_BY).count()
    
    # get days where there were more than NUM_TWEETS_THRESHOLD tweets
    top_dates_df = dates[dates['created_at']>NUM_TWEETS_THRESHOLD]
    if len(top_dates_df):
        print(f"\tDates there were more than {NUM_TWEETS_THRESHOLD} tweets:")
        for i in range(len(top_dates_df)):
            # top date
            top_date = top_dates_df.iloc[[i]].index[0]
            print(f"\t\t{top_date}")
            
            # output to csv
            top_date_output = df[df['date'] == top_date]
            top_date_output.to_csv(f"../dataout/general/{top_date}_tweets.csv")
    else:
        print(f"\tThere are no days where the tweet frequency exeeds {NUM_TWEETS_THRESHOLD} tweets")

    # plot frequency graphs
    plot_frequency_time(dates, overall, selected_topic, data_out, trendline)
    if not overall:
        # plot merged frequency topic graph
        plot_frequency_merge_time()
    print("\tOutput available in dataout/general/")


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
    
    plt.savefig(data_out)
    plt.close()

def plot_frequency_merge_time():
    '''
        Plot sentiment over time for multiple topics.
        Currently set to topics 1, 5, 6, and 7.
    '''
    # get data
    TOPIC_SUBDF_DATA_IN_PREFIX = TOPIC_DATA_IN_PREFIX + "tweet_topic_subdf_topic_"
    data_out = TOPIC_DATA_OUT_PREFIX + f"tweet_frequency_topic_1567.pdf"

    dates1 = prep_df_merged_graph(pd.read_csv(TOPIC_SUBDF_DATA_IN_PREFIX + "1.csv"))
    dates5 = prep_df_merged_graph(pd.read_csv(TOPIC_SUBDF_DATA_IN_PREFIX + "5.csv"))
    dates6 = prep_df_merged_graph(pd.read_csv(TOPIC_SUBDF_DATA_IN_PREFIX + "6.csv"))
    dates7 = prep_df_merged_graph(pd.read_csv(TOPIC_SUBDF_DATA_IN_PREFIX + "7.csv"))

    # set plot lines data
    fig, ax = plt.subplots()
    ax.plot(dates1.index, 'cleaned_tweet', data=dates1, label="Topic 1")
    ax.plot(dates5.index, 'cleaned_tweet', data=dates5, label="Topic 5")
    ax.plot(dates6.index, 'cleaned_tweet', data=dates6, label="Topic 6")
    ax.plot(dates7.index, 'cleaned_tweet', data=dates7, label="Topic 7")

    # Major ticks every month.
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # plot
    plt.title(f'Tweet Frequency of Topics 1, 5, 6 and 7')
    plt.legend(loc="upper left")
    
    # save graph
    plt.savefig(data_out)
    plt.close()

def prep_df_merged_graph(df):
    '''
        Remove unneccessary columns & invalid values, and add date/time columns.
        
        Args:
            df: dataframe to be cleaned.
            dates: cleaned df grouped by date.
    '''
    df = df.drop("Unnamed: 0", axis=1)
    # remove any null created_at values from dataframe
    df = df.drop(df[df['created_at'].isnull()].index)
    # ensure that all values in created_at has 2021 (and not random strings)
    df = df[df['created_at'].str.contains("2021")]

    # split created_at into date and time columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['time'] = df['created_at'].dt.time
    
    dates = df.groupby('date').count()
    return dates

if __name__ == "__main__":
    run(overall=True)
