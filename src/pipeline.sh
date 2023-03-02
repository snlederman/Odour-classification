#!/bin/bash

# this is a pipline running the different scripts in the project

# clean row data
python prep/clean.py

# scale cleaned data
python prep/scale.py

# augment cleaned data
python prep/augment.py --number 100

# augment scaled data
python prep/augment.py --number 100 --scale