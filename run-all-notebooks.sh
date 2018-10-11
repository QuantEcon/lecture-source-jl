#!/bin/bash

red=`tput setaf 1` # For errors
green=`tput setaf 2`
blue=`tput setaf 4`

for f in _build/jupyter/*.ipynb; do
  lec=$(basename $f) # For ease of use
  if [[ $lec =~ .*index.* ]]; then
    echo "${green} skipping index file: $lec"
  else
    echo "${blue} lecture $lec"
    # Check if the target exists, and if it doesn't, add it. 
    if [ -e ./notebooks/"$lec" ]
    then
      echo "${green} target exists; checking timestamps..."
    else 
      echo "${green} target does not exist; creating now."
      \cp "$f" ./notebooks/"$lec" # Move first argument to the right folder. Will overwrite. 
      jupyter nbconvert ./notebooks/"$lec" --ExecutePreprocessor.timeout=900 --to notebook --inplace --allow-errors --execute  # Runs the file.
      echo "${green} conversion complete."
      continue # Move on to the next file. 
    fi 
    # Check if our file is newer than the target. If it is, copy and overwrite. 
    if [[ "$f" -nt ./notebooks/"$lec" ]]
    then
      echo "${green} replacing now"
      \cp "$f" ./notebooks/"$lec" # Move first argument to the right folder. Will overwrite. 
      jupyter nbconvert ./notebooks/"$lec" --ExecutePreprocessor.timeout=600 --to notebook --inplace --allow-errors --execute  # Runs the file. 
      echo "${green} conversion complete"
    else
      echo "${green} target is not older than source."
    fi
  fi
done 