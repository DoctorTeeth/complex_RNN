#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cPickle

"""
The goal is to plot serialized runs after the fact.
What we want is to take as arguments a list of runFiles.
Where a runfile contains the xs (the number of steps)
and ys (the cost at that step) and the label for the graph.

So the one task we have is:
read this out and generate the plot.
"""
def plotter(files):

    for f in files:
        with open(f, 'rb') as pickle_file:
            read_vals = cPickle.load(pickle_file)
            train = read_vals['train_loss']
            test  = read_vals['test_loss']

            if show_test == "test":
                y = test
            else:
                y = train
            x = np.arange(len(y))

            plt.plot(x, y, label=os.path.basename(f))

    plt.xlabel("training sequences")
    plt.ylabel("Cost")
    plt.ylim(ymin=0)
    plt.legend(loc='upper right')
    plt.show()

show_test = sys.argv[1]
files = sys.argv[2:] # all other arguments are results files
print files
plotter(files)
