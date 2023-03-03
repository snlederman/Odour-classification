#!/bin/bash

# this is a pipline running the different scripts in the project

# clean row data
echo clean
python src/prep/clean.py

# scale cleaned data
echo scale
python src/prep/scale.py

# split train test
echo split
python src/prep/split.py

# augment train data
echo augment
python -W ignore src/prep/augment.py --number 100

# scale augmented data
echo scale augment
python -W ignore src/prep/scale.py --augment

