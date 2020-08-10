#!/bin/bash

# Select params.json file
PARMS="./params/params_competition_gefcom2014_load_example.json"

# Extract gefcom2014 data
python ./preprocess/extract_gefcom2014_wind_solar_load.py

# Preprocess gefcom2014 data
python ./preprocess/preprocess_gefcom2014_load_example.py $PARMS

# Train model
python ./main.py $PARMS

# Generate plots
#TODO make a script that generates the results plots