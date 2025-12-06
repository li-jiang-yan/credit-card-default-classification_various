#!/bin/sh
for i in *.py
do
    echo "> pylint $i"
    pylint $i
done
