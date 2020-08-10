#!/bin/bash

# Select params.json file
PARMS="./params/params_competition_gefcom2014_wind0.json"

# Extract gefcom2014 data
python ./preprocess/extract_gefcom2014_wind_solar_load.py

# Preprocess data
python ./preprocess/preprocess_gefcom2014_wind.py $PARMS

# train data
python ./train.py $PARMS
