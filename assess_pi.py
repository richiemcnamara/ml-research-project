import numpy as np

def coverage(test_actual, test_lowers, test_uppers):
    # Returns the % of test observations that were actually covered by their interval
    
    covered = 0
    for i in range(test_actual.shape[0]):
        if (test_actual[i] >= test_lowers[i]) and (test_actual[i] <= test_uppers[i]):
            covered += 1

    return (covered / test_actual.shape[0]) * 100

def width(test_lowers, test_uppers):
    # Returns the mean interval width, min, max, std

    widths = test_uppers - test_lowers

    return widths.mean(), widths.max(), widths.min(), widths.std()