'''
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

Author: Christoph Leitner, Date: Aug. 2022
'''


# Medium Class:
# This class summarizes all properties of the investigated medium (e.g. sizes, speeds, material).
# The class returns a point meshgrid covering the medium in lateral (x) and axial (z) directions.
# Each pixel point is given with the absolut distance (in meter) from the transducer origin.
#
# size of the medium = (type list) = [[1, transducer number of elements], [Number of RX echos (one-way), 1]]
#

import numpy as np

class medium():
    def __init__(self,
                 speed_of_sound_ms=1540,
                 center_frequency=0,
                 sampling_frequency=0,
                 max_depth_wavelength=60,
                 lateral_transducer_element_spacing=None,
                 axial_extrapolation_coef=1,
                 attenuation_coefficient=None,
                 attenuation_power=None):


        self._speed_of_sound = float(speed_of_sound_ms)
        self._center_frequency = center_frequency
        self._sampling_frequency = sampling_frequency

        self._max_penetration_depth_wavelength = self._recorded_penetration_depth = max_depth_wavelength
        self._wavelength_m = self.__wavelength_m()

        self._rx_echo_samples = self.__rx_echo_samples()

        self._lateral_grid_spacing = lateral_transducer_element_spacing
        self._axial_extrapolation_coefficient = axial_extrapolation_coef
        self._imaging_medium = self.__imaging_medium()

        self._attenuation_coefficient = attenuation_coefficient
        self._attenuation_power = attenuation_power


    def __wavelength_m(self):
        us_wavelength = self._speed_of_sound / self._center_frequency
        return us_wavelength


    def __rx_echo_samples(self):
        rx_echo_totalnr_wavelength = (self._recorded_penetration_depth) * 2
        rx_echo_totaltime_s = rx_echo_totalnr_wavelength * self._wavelength_m / self._speed_of_sound
        rx_echo_totalnr_samples = np.round(rx_echo_totaltime_s * self._sampling_frequency)
        return int(rx_echo_totalnr_samples)

    def __imaging_medium(self):
        # EXPLANATION:
        # lateral grid spacing (x) is centered around the center of the transducer
        # lateral spacing is defined by: 1. the number of elements (quantity of points), and
        # 2. the pitch of each element (distance between each point)
        lateral_grid_points_m = self._lateral_grid_spacing
        # EXPLANATION:
        # axial gird spacing (z) is defined by: 1. the number of sampling points (one-direction), and
        # 2. the distance between sampling points: defined by the total length of the travel distance (one-way, = wavelength * number of wavelength)
        # equally shared among all sampling points
        axial_grid_points_m = np.linspace(0,
                                          self._wavelength_m * self._recorded_penetration_depth,
                                          round(self._rx_echo_samples / 2),
                                          endpoint=False)
        imaging_medium_grid = np.meshgrid(lateral_grid_points_m, axial_grid_points_m, sparse=True)
        return imaging_medium_grid

    @property
    def speed_of_sound(self):
        return self._speed_of_sound

    @property
    def rx_echo_totalnr_samples(self):
        return self._rx_echo_samples

    @property
    def recorded_depth(self):
        return self._recorded_penetration_depth

    @property
    def medium(self):
        return self._imaging_medium

    @property
    def alpha(self):
        return self._attenuation_coefficient

    @property
    def alpha_power(self):
        return self._attenuation_power
    
    @property
    def sampling_frequency(self):
        return self._sampling_frequency
    
    @property
    def speed_of_sound(self):
        return self._speed_of_sound

