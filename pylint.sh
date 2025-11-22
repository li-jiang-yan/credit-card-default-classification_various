#!/bin/sh
for i in ann.py logistic_regression.py preprocessing.py random_forest.py svm.py
do
    echo "> pylint $i"
    pylint $i
done
