#!/bin/bash

# Select params.json file
PARMS="./params/params_competition_gefcom2014_solar_example.json"

# Extract gefcom2014 data
python -W ignore ./preprocess/extract_gefcom2014_wind_solar_load.py solar

# Preprocess gefcom2014 data
python -W ignore ./preprocess/preprocess_gefcom2014_solar_example.py $PARMS

# Train model
python -W ignore ./main.py $PARMS

# Generate plots
#TODO make a script that generates the results plots