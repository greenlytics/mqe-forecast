# Global Energy Forecasting Competition 2014 Probabilistic Solar Power Forecasting

Global Energy Forecasting Competition 2014 ([gefcom 2014](https://www.crowdanalytix.com/contests/global-energy-forecasting-competition-2014-probabilistic-solar-power-forecasting)) provides with a rich validation dataset for testing AI and standard solar radiation models. The dataset contains more than 2 years of weather forecast (provided by the [European Centre for Medium-Range Weather Forecast](https://www.ecmwf.int/)) and measurement data with hourly resolution, as well as the details about the equipment at the measurement sites. The main restriction on the data is that it does not provide the geographical location at which the measurements were taken, and lack of such information would normally restricts the use of "clear-sky-alike" solar models for estimating solar radiation at the locations of the solar parks.

## Geographical location of the solar parks
From the measurement data it is possible to estimate the geographical location (latitude and longitude) of the solar parks. The measurement data contains information about the duration of nighttime (without solar production) and daytime (with solar production) and cases provide validation for the implemented models. There are daily and yearly  is compared to the nighttime and daytime duration obtained from the Sun position model where the geographical location is explicitly specified. [A more detailed report](https://github.com/greenlytics/gefcom2014-solar-physical-ai/tree/master/data-location), about the procedure used for estimating the geographical location of the measurement sites, is provided as a part of this repository.

## Physical solar models
Simple atmospheric solar models, based o n the Sun position, have been implemented for estimating the solar radiation on the horizontal surface of the measurement sitesare. Slope and orientation of the solar panels are taken into an account by implementing projection calculations taken from [1]. More advanced, although still a very simple model, slightly modified version of isotropic model was used. See [models](https://github.com/greenlytics/gefcom2014-solar-physical-ai/tree/master/model) for more information about the specific implementation details on models.

## Validation
Test cases provide validation for the implemented models. There is a daily and a yearly test.


## Data description

### Wind track
Data

The target variable is power generation, normalized here by the respective nominal capacities of each wind farm. The explanatory variables that can be used are the past power measurements (if relevant), and input weather forecasts, given as u and v components (zonal and meridional) which can be transformed to wind speed and direction by the contestants, or used readily. These forecasts are given at two heights, 10m and 100m above ground level, in order to give a rough idea of wind profiles.

### Solar track
The target variable is the hourly normalised solar power in MWh/h per MW installed capacity. There are 12 explanatory variables from the [ECMWF](https://www.ecmwf.int/) NWP output to be used as below.


Objective
The topic of the probabilistic solar power forecasting track is to forecast the probabilistic distribution (in quantiles) of the solar power generation for 3 solar farms on a rolling basis. The three solar farms are adjacent. Their installation parameters are:


| | Zone1 | Zone2 | Zone3 |
| --- | :---: | :---: | :---: |
| **Altitude** | 595 m | 602 m | 951 m |
| **Panel type** | Solarfun SF160-24-1M195 | Suntech STP190S-24/Ad+ | Suntech STP200-18/ud |
| **Panel number** | 8 | 26 | 20 |
| **Nominal power** | 1560 Wp | 4940 Wp | 4000 Wp |
| **Orientation** (Clockwise from North) | 38° | 327° | 31° |
| **Tilt** | 36° | 35° | 21° |


| ID | Name | Description | Unit |
| --- | --- | --- | --- |
| 078.128 | Total column liquid water (tclw) | Vertical integral of cloud liquid water content | kg/m^2 |
| 079.128 | Total column ice water (tciw) | Vertical integral of cloud ice water content | kg/m^2 |
| 134.128 | Surface pressure (SP) | Pressure at earth surface | Pa |
| 157.128 | Relative humidity at 1000 mbar (r) | Relative humidity is defined with respect to saturation of the mixed phase | % |
| 164.128 | Total cloud cover (TCC) | Total cloud cover derived from model levels using the model's overlap assumption | - |
| 165.128 | U wind component (10U) | Longitudinal wind component at 10 meters height | m/s |
| 166.128 | V wind component (10V) | Latitudinal wind component at 10 meters height | m/s |
| 167.128 | Temperature (2T) | Temperature at 2 meters height | K |
| 169.128 | Surface solar rad down (SSRD) | - | J/m^2 Accumulated field |
| 175.128 | Surface thermal rad down (STRD) | - | J/m^2 Accumulated field |
| 178.128 | Top net solar rad (TSR) | Net solar radiation at the top of the atmosphere | J/m^2 Accumulated field |
| 228.128 | Total precipitation (TP) | Convective precipitation + stratiform precipitation (CP +LSP) | m Accumulated field |