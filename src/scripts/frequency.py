'''
    Plot tweet frequency over time for large dataset
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# file paths
OVERALL_DATA_IN = "../datain/topic_modelling/cleaned_tweets_largest_community.csv" # overall tweets
OVERALL_DATA_OUT = "../dataout/general/tweet_frequency_overall.jpeg" # overall tweets

LARGEST_TOPIC_DATA_IN = "../datain/topic_modelling/cleaned_tweets_largest_topic.csv" # largest topic
LARGEST_TOPIC_DATA_OUT = "../dataout/general/tweet_frequency_topic_11.jpeg" # largest topic

# file paths for sample data
# SAMPLE_DATA_IN = "../datain/topic_modelling/cleaned_tweets.csv"
# SAMPLE_DATA_OUT = "../dataout/general/Total_tweet_frequency.jpeg"

def run(topic_position=0, overall=False):
    '''
        Run frequency code.
        Default run for largest topic.

        Args:
            topic_position: int (0 for largest topic, 1 for second largest, etc.)
            overall: boolean (true if want to analyse overall data frequency, false if not)
    '''
    if not overall:
        print("Running selected topic tweet frequency")
        print("\tSetting topic I/O files...")
        if topic_position == 0:
            data_in = LARGEST_TOPIC_DATA_IN
            data_out = LARGEST_TOPIC_DATA_OUT
        else:
            # TODO: Allow for different topic input/output
            print("TODO: Have not set non-largest topic input/output files yet")
            return
    else:
        print("Running overall tweet frequency")
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
    #https://intellipaat.com/community/13909/python-how-can-i-split-a-column-with-both-date-and-time-e-g-2019-07-02-00-12-32-utc-into-two-separate-columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['time'] = df['created_at'].dt.time

    # group tweets by date and count number of entries per day
    dates = df.groupby('date').count()
    plot_frequency_time(dates, overall, data_out)
    print("\tGraph outputs available in dataout/")


def plot_frequency_time(dates, overall, data_out):
    '''
        Plot tweet frequency over time.

        Args:
            dates: df with count of number of tweets posted grouped by date
            overall: True if is for an overall analysis, False if it is for a topic's analysis.
            data_out: file to which the plot output should be written to.
    '''
    fig, ax = plt.subplots()
    ax.plot(dates.index, 'cleaned_tweet', data=dates)
    # Major ticks every 6 months.
    fmt_half_year = mdates.MonthLocator(interval=1)
    ax.xaxis.set_major_locator(fmt_half_year)
    # plot
    if overall:
        plt.title('Overall Tweet Frequency over time: 1 Feb - 31 May')
    else:
        # TODO: edit this such that it can be any topic
        plt.title('Largest Topic Tweet Frequency over time: 1 Feb - 31 May')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.savefig(data_out)
    plt.close()

if __name__ == "__main__":
    run()
