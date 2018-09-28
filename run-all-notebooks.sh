#!/bin/bash

redln() { echo "$(tput setaf 1)$*$(tput setaf 9)"; } # Prints in red. For error messages. 

for f in _build/jupyter/*.ipynb; do
  lec=$(basename $f) # For ease of use 
  # Check if the target exists, and if it doesn't, add it. 
  if [ -e ./notebooks/"$lec" ]
  then
    :
  else 
    echo "target does not exist; creating now."
    \cp "$f" ./notebooks/"$lec" # Move first argument to the right folder. Will overwrite. 
    jupyter nbconvert ./notebooks/"$lec" --ExecutePreprocessor.timeout=600 --to notebook --inplace --allow-errors --execute  # Runs the file. 
    continue # Move on to the next file. 
  fi 
  # Check if our file is newer than the target. If it is, copy and overwrite. 
  if [["$f" -nt ./notebooks/"$lec"]]
  then
    \cp "$f" ./notebooks/"$lec" # Move first argument to the right folder. Will overwrite. 
    jupyter nbconvert ./notebooks/"$lec" --ExecutePreprocessor.timeout=600 --to notebook --inplace --allow-errors --execute  # Runs the file. 
  fi 
done 