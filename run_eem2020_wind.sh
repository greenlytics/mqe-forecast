#!/bin/bash

# Select params.json file
PARMS=${1:-'./params/params_eem2020_wind_example.json'}

# Preprocess EEM2020 data
python -W ignore ./preprocess/preprocess_eem2020_wind_example.py $PARMS

# Train model
python -W ignore ./gbdt_forecast.py $PARMS

# Generate plots
# python -W ignore ./generate_plots_wind.py
