#!/bin/bash

filename = $1 # Grab our input. 
\cp ./_build/jupyter/filename ./notebooks/filename # Move it to the right folder. Will overwrite. 
jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute filename # Runs the file. 