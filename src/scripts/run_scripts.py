'''
    Run all cleaning, topic modelling, and sentiment analysis scripts.
    TODO: Does not run elbow method, since it causes the plots to include the elbow method plot in output.
'''
import frequency, clean_corpus_for_btm, clean_corpus_for_sentiment # general
import one_topic_flow, elbow_method # topic modelling
import sentiment_segments # sentiment analysis

import os # creating directories

topic_position = 0 # 0 is the largest topic, 1 second largest, etc.

def create_output_directories():
    '''
        Create output directories for scripts if they don't already exist.
    '''
    if not os.path.exists('../dataout/general'):
        os.makedirs('../dataout/general')
    if not os.path.exists('../dataout/sentiment'):
        os.makedirs('../dataout/sentiment')
    if not os.path.exists('../dataout/topic_modelling'):
        os.makedirs('../dataout/topic_modelling')

if __name__ == "__main__":
    create_output_directories()

    # topic modelling & sentiment cleaning (only clean if haven't cleaned before)
    if not os.path.exists('../datain/clean'):
        clean_corpus_for_btm.run()
        clean_corpus_for_sentiment.run()
    else:
        print("Skipping cleaning cleaning step (already cleaned data a previous time)")

    # tweet frequency
    frequency.run(overall=True)
    frequency.run(topic_position=topic_position)

    # topic modelling & sentiment per segment
    one_topic_flow.run()

    # sentiment analysis (segments)
    sentiment_segments.run()
