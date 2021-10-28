'''
Plot tweet frequency over time for large dataset
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# file paths
# FREQUENCY_INPUT_FILE = "../datain/topic_modelling/cleaned_tweets_largest_community.csv"
# FREQUENCY_OUTPUT_FILE = "../dataout/general/Total_tweet_frequency_largest_community.jpeg"

# file paths for sample data
# FREQUENCY_INPUT_FILE = "../datain/topic_modelling/cleaned_tweets.csv"
# FREQUENCY_OUTPUT_FILE = "../dataout/general/Total_tweet_frequency.jpeg"

DATA_IN = "../datain/topic_modelling/cleaned_tweets_largest_community.csv" # overall tweets
# DATA_IN = "../datain/topic_modelling/cleaned_tweets_largest_topic.csv" # largest topic

DATA_OUT = "../dataout/general/overall_tweet_frequency.jpeg" # overall tweets
# DATA_OUT = "../dataout/general/largest_topic_tweet_frequency.jpeg" # largest topic

def run():
    print("Running tweet frequency")
    # load tweet corpus data
    df = pd.read_csv(DATA_IN)
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
    plot_frequency_time(dates)
    print("Finished running tweet frequency")


def plot_frequency_time(dates):
    '''
    Plot tweet frequency over time
    @param dates - df with count of number of tweets posted grouped by date
    '''
    fig, ax = plt.subplots()
    ax.plot(dates.index, 'cleaned_tweet', data=dates)
    # Major ticks every 6 months.
    fmt_half_year = mdates.MonthLocator(interval=1)
    ax.xaxis.set_major_locator(fmt_half_year)
    # plot
    plt.title('Largest Community Tweet Frequency over time: 1 Feb - 31 May')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.savefig(DATA_OUT)
    plt.close()

if __name__ == "__main__":
    run()
