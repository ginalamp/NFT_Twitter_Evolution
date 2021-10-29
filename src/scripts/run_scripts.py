'''
Run all cleaning, topic modelling, and sentiment analysis scripts
'''
import frequency, clean_corpus_for_btm, clean_corpus_for_sentiment # general
import one_topic_flow, elbow_method # topic modelling
import sentiment_segments # sentiment analysis

import os # creating directories

overall = True
topic = 1 # 1 is the largest topic, 2 second largest, etc.

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

    # topic modelling & sentiment cleaning
    clean_corpus_for_btm.run()
    clean_corpus_for_sentiment.run()

    # tweet frequency
    frequency.run(topic, overall)
    frequency.run(topic, not overall)

    # topic modelling & sentiment per segment
    one_topic_flow.run()
    elbow_method.run() # TODO: doesn't currently output anything

    # sentiment analysis (segments)
    sentiment_segments.run()

    # old.topic_modelling.run() # old
    # sentiment_analysis.run() # old
