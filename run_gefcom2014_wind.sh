#!/bin/bash

# Select params.json file
PARMS=${1:-'./params/params_gefcom2014_wind_example.json'}

# Extract gefcom2014 data
python -W ignore ./preprocess/extract_gefcom2014_wind_solar_load.py wind

# Preprocess gefcom2014 data
python -W ignore ./preprocess/preprocess_gefcom2014_wind_example.py $PARMS

# Train model
python -W ignore ./gbdt_forecast.py $PARMS

# Generate plots
# python -W ignore ./generate_plots_wind.py
