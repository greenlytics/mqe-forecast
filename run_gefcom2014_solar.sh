#!/bin/bash

# Select params.json file
PARMS="./params/params_gefcom2014_solar_competition.json"

# Extract gefcom2014 data
#python -W ignore ./preprocess/extract_gefcom2014_wind_solar_load.py solar

# Preprocess gefcom2014 data
#python -W ignore ./preprocess/preprocess_gefcom2014_solar_example.py $PARMS

# Train model
python -W ignore ./gbdt_forecast.py $PARMS

# Generate plots
#python -W ignore ./plots/generate_plots_solar.py