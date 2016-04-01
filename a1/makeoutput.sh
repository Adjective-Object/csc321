#!/usr/bin/env bash

mkdir -p output
# find the best k
python knn.py default_dataset.json validation 1 10

# use the best k on the test set
python knn.py default_dataset.json test 2 2

# now run on the other actors
python knn.py different_actors_dataset.json test 2 2

