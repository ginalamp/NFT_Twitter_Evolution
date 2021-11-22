'''
    Run all cleaning, frequency, topic modelling, and sentiment analysis scripts.

    NOTE: Does not run elbow method, since it causes the plots to include the elbow method plot in output.
'''
import frequency, clean_corpus # general
import single_topic_analysis, elbow_method # topic modelling
import sentiment_segments # sentiment analysis

import os # creating directories

# command line arguments
import getopt, sys 
argumentList = sys.argv[1:]
options = "hcs:" # help, clean, sample
long_options = ["help", "clean", "sample"]

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

    # command line arguments
    force_clean = False
    run_sample = False
    try:
        # Parsing argument
        arguments, _ = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument in arguments:
            if currentArgument in ("-h", "--help"):
                print("Displaying Help")
                exit()
            elif currentArgument in ("-c", "--clean"):
                print("Force clean data:")
                force_clean = True
            elif currentArgument in ("-s", "--sample"):
                print("Run sample data (instead of full dataset) -- Not implemented.")
                run_sample = True
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    # topic modelling & sentiment cleaning (forced or only clean if haven't cleaned before)
    sentiment_cleaned_path = '../datain/sentiment/cleaned_tweets_for_sentiment.csv'
    btm_cleaned_path = '../datain/topic_modelling/cleaned_tweets_largest_community_btm.csv'
    freq_cleaned_path = '../datain/topic_modelling/cleaned_tweets_largest_community.csv'
    if force_clean or not (os.path.exists(sentiment_cleaned_path) and os.path.exists(btm_cleaned_path) and os.path.exists(freq_cleaned_path)):
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
