"""
parsing arguments from command line
"""

import argparse

# argparse function
def get_args(argv=None):
    """
    Takes arguments from the user when running as a command line script
    """
    parser = argparse.ArgumentParser(description="define whether to work on scaled data")
    parser.add_argument("-s","--scale", action="store_true", help="standard scaling")
    parser.add_argument("-a","--augment", action="store_true", help="augmented data")
    parser.add_argument("-n","--number", type=int, help="repeats the action n times")
    parser.add_argument("-f","--fourier", action="store_true", help="fourier transform data")
    return vars(parser.parse_args(argv))
