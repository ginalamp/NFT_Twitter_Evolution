'''
    Get optimal number of topics based on BTM logLik value output (R BTM functions).
    
    NOTE: This is not included in the run_scropts.py, since it messes with the other plot outputs.
'''

import matplotlib.pyplot as plt
from kneed import KneeLocator # elbow method
import pandas as pd
import numpy as np
# from scipy.interpolate import interp1d # normalise curve

# NB: Need to make sure that the input LogLik column does not have any commas in the number.
DATA_IN = '../datain/topic_modelling/ElbowMethodData.csv' # overall data LogLik values for largest community

def run():
    '''
        Run functions for elbow method for overall data.
        Uses BTM R function LogLik output (added manually to a csv).
    '''
    print("Applying elbow method...")
    # get data
    df = pd.read_csv(DATA_IN)

    # convert columns to numpy 2D arrays
    topics = df[['Topics']].to_numpy()
    loglikvals = df[['LogLik']].to_numpy()

    # convert 2D numpy array to 1D numpy array
    x = np.concatenate(topics).ravel()
    y = np.concatenate(loglikvals).ravel()

    # apply kneelocator method
    optimal_topics = plot_knee_not_normalised(x, y)
    print("\tOptimal amount of topics (not normalised):", optimal_topics.elbow)

    optimal_topics = plot_knee_normalised(x, y)
    print("\tOptimal amount of topics (normalised):", optimal_topics.elbow)

def plot_knee_not_normalised(x, y):
    '''
        Apply non-normalised knee method.

        Args:
            x: numpy 1D array with the number of topics.
            y: numpy 1D array with the LogLik value associated with the number of topics.
        Returns:
            optimal_num_topics.elbow: Most optimal point.
    '''
    optimal_num_topics = KneeLocator(x, y, curve="concave", direction="increasing")
    optimal_num_topics.plot_knee_normalized()
    return optimal_num_topics

def plot_knee_normalised(x, y):
    '''
        Apply default polynomial knee method normalisation.

        Args:
            x: numpy 1D array with the number of topics.
            y: numpy 1D array with the LogLik value associated with the number of topics.
        Returns:
            optimal_num_topics.elbow: Most optimal point.
    '''
    optimal_num_topics = KneeLocator(x, y, curve="concave", direction="increasing", interp_method="polynomial")
    optimal_num_topics.plot_knee_normalized()
    return optimal_num_topics

if __name__ == "__main__":
    run()