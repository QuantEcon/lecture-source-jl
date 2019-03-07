#!/bin/bash

red=`tput setaf 1` # For errors
green=`tput setaf 2`
blue=`tput setaf 4`

make clean
make jupyter

for f in $(find _build/jupyter -name "*.ipynb"); do
  lec=$(basename $f) # For ease of use
  if [[ $lec =~ .*index.* ]]; then
    echo "${green} skipping index file: $lec"
  else
    jupyter nbconvert $f --ExecutePreprocessor.timeout=900 --to notebook --inplace --allow-errors --execute
  fi
done
