'''
    Get optimal number of topics based on BTM logLik value output (R BTM functions).
'''

import matplotlib.pyplot as plt
from kneed import KneeLocator # elbow method
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d # normalise curve

# NB: Need to make sure that the input LogLik column does not have any commas in the number.
DATA_IN = '../datain/topic_modelling/ElbowMethodData.csv' # overall data LogLik values for largest community

def run():
    '''
        Run functions for elbow method for overall data.
        Uses BTM R function LogLik output (added manually to a csv).

        Returns:
            optimal_num_topics: Optimal number of topics identified by the elbow method
    '''
    print("Running elbow method")
    # get data
    df = pd.read_csv(DATA_IN)

    # convert columns to numpy 2D arrays
    topics = df[['Topics']].to_numpy()
    loglikvals = df[['LogLik']].to_numpy()

    # convert 2D numpy array to 1D numpy array
    x = np.concatenate(topics).ravel()
    y = np.concatenate(loglikvals).ravel()

    # apply kneelocator method
    optimal_num_topics = plot_knee_not_normalised(x, y)
    print("\tOptimal amount of topics (not normalised):", optimal_num_topics.elbow)

    optimal_num_topics = plot_knee_normalised(x, y)
    print("\tOptimal amount of topics (normalised):", optimal_num_topics.elbow)
    return optimal_num_topics

def plot_knee_not_normalised(x, y):
    '''
        Apply non-normalised knee method.

        Args:
            x:
            y:
        Returns:
            optimal_num_topics: Most optimal point.
    '''
    optimal_num_topics = KneeLocator(x, y, curve="concave", direction="increasing")
    optimal_num_topics.plot_knee_normalized()
    return optimal_num_topics

def plot_knee_normalised(x, y):
    '''
        Apply default polynomial knee method normalisation.

        Args:
            x:
            y:
        Returns:
            optimal_num_topics: Most optimal point.
    '''
    optimal_num_topics = KneeLocator(x, y, curve="concave", direction="increasing", interp_method="polynomial")
    optimal_num_topics.plot_knee_normalized()
    return optimal_num_topics

if __name__ == "__main__":
    run()