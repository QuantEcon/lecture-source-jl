#!/bin/bash

redln() { echo -n "$(tput setaf 1)$*$(tput setaf 9)"; } # Prints in red. For error messages. 
if [ $# -eq 0 ]; then
    redln "No arguments supplied. Will exit now."
    exit 1
fi

# Otherwise. 
\cp ./_build/jupyter/$1 ./notebooks/$1 # Move first argument to the right folder. Will overwrite. 
jupyter nbconvert ./notebooks/$1 --ExecutePreprocessor.timeout=600 --to notebook --inplace --allow-errors --execute  # Runs the file. 