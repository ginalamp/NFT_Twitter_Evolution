'''
Run all cleaning, topic modelling, and sentiment analysis scripts
'''

import clean_corpus
import topic_modelling
import sentiment_analysis

import os # creating directories

def create_output_directories():
    '''
    Create output directories for scripts if they don't already exist.
    '''
    if not os.path.exists('dataout/sentiment'):
        os.makedirs('dataout/sentiment')
    if not os.path.exists('dataout/topic_modelling'):
        os.makedirs('dataout/topic_modelling')

if __name__ == "__main__":
    create_output_directories()
    clean_corpus.run()
    topic_modelling.run()
    sentiment_analysis.run()
