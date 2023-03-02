#!/bin/bash

# this is a pipline running the different scripts in the project

# clean row data
python src/prep/clean.py

# scale cleaned data
python src/prep/scale.py

# augment cleaned data
python src/prep/augment.py --number 100

# augment scaled data
python src/prep/augment.py --number 100 --scale