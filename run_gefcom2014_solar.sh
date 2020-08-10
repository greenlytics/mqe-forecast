#!/bin/bash

# Select params.json file
PARMS="./params/params_competition_gefcom2014_solar.json"

# Prepare gefcom2014 data
python ./preprocess/prepare_gefcom2014_wind_solar.py

#           echo $STR
# Preprocess data
#python ./preprocess/preprocess_gefcom2014_solar.py $PARMS
