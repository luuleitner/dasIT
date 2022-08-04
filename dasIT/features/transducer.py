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

# Transducer class:
# This class summarizes all necessary properties of the transducer that are needed for the reconstruction of images.

import numpy as np

class transducer():
    def __init__(self,
                 center_frequency_hz=0,
                 bandwidth_hz=0,
                 adc_ratio=1,
                 transducer_elements_nr=1,
                 element_pitch_m=None,
                 pinmap=None,
                 pinmapbase=1,
                 elevation_focus=None,
                 focus_number=None,
                 totalnr_planewaves=1,
                 planewave_angle_interval=[0,0],
                 axial_cutoff_wavelength=5,
                 speed_of_sound_ms=None):

        self._f_center = float(center_frequency_hz)
        self._bandwidth = bandwidth_hz
        self._samples_per_wavelength = adc_ratio
        self._f_sampling = self._samples_per_wavelength * self._f_center
        self._transducer_elements = int(transducer_elements_nr)
        self._element_pitch = float(element_pitch_m)
        self._lateral_transducer_spacing = self.lateral_spacing()
        self._pw_active_aperture = self._element_pitch * self._transducer_elements
        self._focus_number = focus_number
        self._elevation_focus = elevation_focus
        self._nr_planewaves = int(totalnr_planewaves)
        self._planewave_interval = planewave_angle_interval
        self._planwave_angles = self.planewave_angles()
        self._initial_axial_cutoff_wavelength = axial_cutoff_wavelength


        if (pinmap is not None) and pinmapbase == 1:
            self._pinmap = pinmap - 1
        elif (pinmap is not None) and pinmapbase == 0:
            self._pinmap = pinmap
        else:
            self._pinmap = np.linspace[1, self._transducer_elements, 1]

        self._speed_of_sound = speed_of_sound_ms
        self._wavelength = self.wavelength_m()
        self._start_depth_samples = self.start_depth_recording_samples()
        self._start_depth_m = self.start_depth_recording_m()


    def planewave_angles(self):
        angles = np.linspace(self._planewave_interval[0], self._planewave_interval[1], self._nr_planewaves)
        angles = np.radians(angles)
        return angles.reshape(-1,1)


    # The lateral spacing function sets the center of the transducer on the origin of the coordinate system (x,z)
    def lateral_spacing(self):
        spacing_around_zero = np.linspace((self._element_pitch * self._transducer_elements / 2) * -1,
                                          (self._element_pitch * self._transducer_elements / 2),
                                          self._transducer_elements)
        return spacing_around_zero


    def wavelength_m(self):
        us_wavelength = self._speed_of_sound / self._f_center
        return us_wavelength


    def start_depth_recording_samples(self):
        start_depth = np.round((((self._initial_axial_cutoff_wavelength * 2) * self._wavelength) / self._speed_of_sound) * self._f_sampling)
        return int(start_depth)


    def start_depth_recording_m(self):
        start_depth = ((self._initial_axial_cutoff_wavelength * 2) * self._wavelength)
        return start_depth


    @property
    def center_frequency(self):
        return self._f_center

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def sampling_frequency(self):
        return self._f_sampling

    @property
    def samples_per_wavelength(self):
        return self._samples_per_wavelength

    @property
    def transducer_elements(self):
        return self._transducer_elements

    @property
    def element_pitch(self):
        return self._element_pitch

    @property
    def lateral_transducer_spacing(self):
        return self._lateral_transducer_spacing

    @property
    def pw_aperture(self):
        return self._pw_active_aperture

    @property
    def transducer_pinmap(self):
        return self._pinmap

    @property
    def fnumber(self):
        return self._focus_number

    @property
    def elevation_focus(self):
        return self._elevation_focus

    @property
    def planewaves_nr(self):
        return self._nr_planewaves

    @property
    def planewave_angles_intervall(self):
        return self._planwave_angles

    @property
    def start_depth_rec_samples(self):
        return self._start_depth_samples

    @property
    def start_depth_rec_m(self):
        return self._start_depth_m