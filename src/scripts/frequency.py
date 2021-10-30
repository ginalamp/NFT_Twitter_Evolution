'''
    Plot tweet frequency over time for large dataset
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# file paths
OVERALL_DATA_IN = "../datain/topic_modelling/cleaned_tweets_largest_community.csv" # overall tweets
OVERALL_DATA_OUT = "../dataout/general/tweet_frequency_overall.jpeg" # overall tweets

TOPIC_DATA_IN_PREFIX = "../datain/topic_modelling/" # topic
TOPIC_DATA_OUT_PREFIX = "../dataout/general/" # topic

# file paths for sample data
# SAMPLE_DATA_IN = "../datain/topic_modelling/cleaned_tweets.csv"
# SAMPLE_DATA_OUT = "../dataout/general/Total_tweet_frequency.jpeg"

def run(overall=False, selected_topic=11):
    '''
        Run frequency code.
        Default run for largest topic.

        Args:
            overall: boolean (true if want to analyse overall data frequency, false if not)
            selected_topic: the selected topic number
    '''
    if not overall:
        print("Calculating selected topic tweet frequency...")
        print("\tSetting topic I/O files...")
        data_in = TOPIC_DATA_IN_PREFIX + f"tweet_sentiment_subdf_topic_{selected_topic}.csv"
        data_out = TOPIC_DATA_OUT_PREFIX + f"tweet_frequency_topic_{selected_topic}.jpeg"
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
    #https://intellipaat.com/community/13909/python-how-can-i-split-a-column-with-both-date-and-time-e-g-2019-07-02-00-12-32-utc-into-two-separate-columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['time'] = df['created_at'].dt.time

    # group tweets by date and count number of entries per day
    dates = df.groupby('date').count()
    plot_frequency_time(dates, overall, selected_topic, data_out)
    print("\tGraph outputs available in dataout/")


def plot_frequency_time(dates, overall, selected_topic, data_out):
    '''
        Plot tweet frequency over time.

        Args:
            dates: df with count of number of tweets posted grouped by date
            overall: True if is for an overall analysis, False if it is for a topic's analysis.
            selected_topic: the selected topic number
            data_out: path to the file to which this function will output to.
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
        plt.title(f'Topic {selected_topic} Tweet Frequency over time: 1 Feb - 31 May')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.savefig(data_out)
    plt.close()

if __name__ == "__main__":
    run()
