'''
    Run all cleaning, frequency, topic modelling, and sentiment analysis scripts.

    NOTE: Does not run elbow method, since it causes the plots to include the elbow method plot in output.
'''
import frequency, clean_corpus # general
import single_topic_analysis, elbow_method # topic modelling
import sentiment_segments # sentiment analysis

import os # creating directories

topic_position = 0 # 0 is the largest topic, 1 second largest, etc.

def create_output_directories():
    '''
        Create output directories for scripts if they don't already exist.
    '''
    # final output directories
    if not os.path.exists('../dataout/general'):
        os.makedirs('../dataout/general')
    if not os.path.exists('../dataout/sentiment'):
        os.makedirs('../dataout/sentiment')
    if not os.path.exists('../dataout/topic_modelling'):
        os.makedirs('../dataout/topic_modelling')
    # in process output directories
    if not os.path.exists('../datain/clean'):
        os.makedirs('../datain/clean')
    if not os.path.exists('../datain/sentiment'):
        os.makedirs('../datain/sentiment')
    if not os.path.exists('../datain/topic_modelling'):
        os.makedirs('../datain/topic_modelling')

if __name__ == "__main__":
    # create output directories
    create_output_directories()

    # topic modelling & sentiment cleaning (only clean if haven't cleaned before)
    sentiment_cleaned_path = '../datain/sentiment/cleaned_tweets_for_sentiment.csv'
    btm_cleaned_path = '../datain/topic_modelling/cleaned_tweets_largest_community_btm.csv'
    freq_cleaned_path = '../datain/topic_modelling/cleaned_tweets_largest_community.csv'
    if not (os.path.exists(sentiment_cleaned_path) and os.path.exists(btm_cleaned_path) and os.path.exists(freq_cleaned_path)):
        clean_corpus.run()
    else:
        print("Skipping cleaning cleaning step (already cleaned data in previous run)")

    # topic modelling & sentiment per segment
    # elbow_method.run() # NOTE: this messes with topic analysis graph output
    selected_topic = single_topic_analysis.run()

    # sentiment analysis (segments)
    sentiment_segments.run()

    # tweet frequency
    frequency.run(overall=True)
    print(f"Running frequency for selected topid {selected_topic}")
    frequency.run(selected_topic=selected_topic)
