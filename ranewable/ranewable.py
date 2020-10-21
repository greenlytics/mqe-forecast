import numpy as np
import pandas as pd

import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.forecast import ForecastModel

class Ra():

    def __init__(self,
                 longitude,
                 latitude,
                 altitude,
                 capacity,
                 orientation,
                 tilt,
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model='pvwatts',
                 ac_model='pvwatts',
                 aoi_model='no_loss',
                 spectral_model='no_loss',
                 temperature_model='sapm',
                 losses_model='no_loss'):

        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.capacity = capacity
        self.orientation = orientation
        self.tilt = tilt

        self.location = Location(longitude=self.longitude,
                                 latitude=self.latitude,
                                 tz='Europe/Berlin',
                                 altitude=self.altitude)
        self.system = PVSystem(surface_tilt=self.tilt,
                               surface_azimuth=self.orientation,
                               module_parameters={'pdc0': capacity, 'gamma_pdc': -0.004},
                               inverter_parameters={'pdc0': capacity})
        self.model = ForecastModel(model_type=None, model_name=None, set_type=None)
        self.model.location = self.location

        self.set_modelchain(clearsky_model=clearsky_model,
                          transposition_model=transposition_model,
                          solar_position_method=solar_position_method,
                          airmass_model=airmass_model,
                          dc_model=dc_model,
                          ac_model=ac_model,
                          aoi_model=aoi_model,
                          spectral_model=spectral_model,
                          temperature_model=temperature_model,
                          losses_model=losses_model)


    def set_modelchain(self,
                       clearsky_model='ineichen',
                       transposition_model='haydavies',
                       solar_position_method='nrel_numpy',
                       airmass_model='kastenyoung1989',
                       dc_model='pvwatts',
                       ac_model='pvwatts',
                       aoi_model='no_loss',
                       spectral_model='no_loss',
                       temperature_model='sapm',
                       losses_model='no_loss'):

        self.modelchain = ModelChain(self.system,
                                     self.location,
                                     clearsky_model=clearsky_model,
                                     transposition_model=transposition_model,
                                     solar_position_method=solar_position_method,
                                     airmass_model=airmass_model,
                                     dc_model=dc_model,
                                     ac_model=ac_model,
                                     aoi_model=aoi_model,
                                     spectral_model=spectral_model,
                                     temperature_model=temperature_model,
                                     losses_model=losses_model)




    def infer_midx(self, index): 
        is_multi_index = isinstance(index, pd.MultiIndex)
        if is_multi_index:
            multi_index = index
            index = index.get_level_values(1)

        else: 
            multi_index = None
        
        return index, is_multi_index, multi_index


    def infer_freq_midx(self, multi_index): 
        # To infer frequency it has to be same everywhere (discernible frequency). This is not true
        # if multiindex with (ref_datetime, valid_datetime) and overlapping valid_datetimes. Therefore, 
        # we use only the first ref_datetime to infer the frequency of all valid_datetimes. 

        idx_first_ref_datetime = multi_index.get_level_values(0) == multi_index.get_level_values(0)[0]
        first_valid_datetimes = multi_index[idx_first_ref_datetime].get_level_values(1)
        freq_str = pd.infer_freq(first_valid_datetimes)

        return freq_str

    def calculate_solpos(self, index, freq_str=None):

        # Infer multiindex 
        index, is_multi_index, multi_index = self.infer_midx(index)

        if freq_str == None: 
            # Infer frequency
            if is_multi_index: 
                freq_str = self.infer_freq_midx(multi_index)
            else:
                freq_str = pd.infer_freq(index)

        # Use middle point in interval as estimate for the mean
        offset_datetime = pd.tseries.frequencies.to_offset(freq_str).delta/2
        solpos = self.location.get_solarposition(index+offset_datetime)

        if is_multi_index:
            solpos.index = multi_index
        else:
            solpos.index = index

        return solpos

    def calculate_clearsky(self, index, freq_str=None):

        # Infer multiindex 
        index, is_multi_index, multi_index = self.infer_midx(index)

        if freq_str == None: 
            # Infer frequency
            if is_multi_index: 
                freq_str = self.infer_freq_midx(multi_index)
            else:
                freq_str = pd.infer_freq(index)

        # Use middle point in interval as estimate for the mean
        offset_datetime = pd.tseries.frequencies.to_offset(freq_str).delta/2
        clearsky = self.location.get_clearsky(index+offset_datetime)
        clearsky.index = clearsky.index-offset_datetime

        if is_multi_index:
            clearsky.index = multi_index

        return clearsky


    def weather_from_tcc(self, tcc, freq_str=None):

        # Infer multiindex 
        index, is_multi_index, multi_index = self.infer_midx(tcc.index)
        tcc.index = index

        if freq_str == None: 
            # Infer frequency
            if is_multi_index: 
                freq_str = self.infer_freq_midx(multi_index)
            else:
                freq_str = pd.infer_freq(index)

        weather = self.model.cloud_cover_to_irradiance(tcc, how='clearsky_scaling')

        if is_multi_index:
            weather.index = multi_index

        return weather


    def weather_from_ghi(self, ghi, freq_str=None):

        # Infer multiindex 
        index, is_multi_index, multi_index = self.infer_midx(ghi.index)
        ghi.index = index

        if freq_str == None: 
            # Infer frequency
            if is_multi_index: 
                freq_str = self.infer_freq_midx(multi_index)
            else:
                freq_str = pd.infer_freq(index)

        solpos = self.calculate_solpos(ghi.index, freq_str=freq_str)
        weather = pvlib.irradiance.erbs(ghi, solpos['zenith'], ghi.index)
        weather['ghi'] = ghi

        if is_multi_index:
            weather.index = multi_index

        return weather


    def calculate_power_clearsky(self, index, freq_str=None):
        
        # Infer multiindex
        index, is_multi_index, multi_index = self.infer_midx(index)

        if freq_str == None: 
            # Infer frequency
            if is_multi_index:
                freq_str = self.infer_freq_midx(multi_index)
            else:
                freq_str = pd.infer_freq(index)

        clearsky = self.calculate_clearsky(index, freq_str=freq_str)
        power_clearsky = self.modelchain.run_model(times=clearsky.index, weather=clearsky).dc

        if is_multi_index:
            power_clearsky.index = multi_index

        return power_clearsky


    def calculate_power(self, weather, freq_str=None):

        # Infer multiindex 
        index, is_multi_index, multi_index = self.infer_midx(weather.index)
        weather.index = index

        if freq_str == None: 
            # Infer frequency
            if is_multi_index: 
                freq_str = self.infer_freq_midx(multi_index)
            else:
                freq_str = pd.infer_freq(index)

        power_clearsky = self.calculate_power_clearsky(weather.index, freq_str=freq_str)
        power = self.modelchain.run_model(times=weather.index, weather=weather).dc

        # If power is greater than clearsky use clearsky
        idx_clearsky = power > power_clearsky
        power.loc[idx_clearsky] = power_clearsky.loc[idx_clearsky]

        # If power is negative set to zero
        idx_negative = power < 0
        power.loc[idx_negative] = 0

        if is_multi_index:
            power.index = multi_index

        return power
