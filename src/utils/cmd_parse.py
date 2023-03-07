"""
parsing arguments from command line
"""

import os
import argparse

# argparse function
def get_args(argv=None):
    """
    Takes arguments from the user when running as a command line script
    """
    parser = argparse.ArgumentParser(description="define whether to work on scaled data")
    parser.add_argument("-s","--scaled", action="store_true", help="standard scaling")
    parser.add_argument("-a","--augmented", action="store_true", help="augmented data")
    parser.add_argument("-c","--clipped", action="store_true", help="clipped data")
    parser.add_argument("-n","--number", type=int, help="repeats the action n times")
    return vars(parser.parse_args(argv))

def args_to_path(args):
    """takes args dict and return a string path"""
    path_dict = dict()
    for key,value in args.items():
        if value:
            path_dict[key] = key
        else:
            path_dict[key] = ""
    return os.path.join(path_dict["augmented"], path_dict["scaled"], path_dict["clipped"])
    