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


import numpy as np
from scipy.interpolate import RectBivariateSpline

class interpolate_bf():
    def __init__(self, signals=None, transducer=None, medium=None, axial_scale=1, lateral_scale=2):
        self._signals = signals
        self._axial_interpolation_factor = axial_scale
        self._lateral_interpolation_factor = lateral_scale
        self._active_aperture_size = transducer._pw_active_aperture
        self._recorded_depth = medium.recorded_depth * transducer.wavelength

        self._axial_spacing_idx = np.arange(0, self._signals.shape[0], 1)
        self._axial_spacing = self._signals.shape[0]
        self._lateral_spacing_idx = np.arange(0, self._signals.shape[1], 1)
        self._lateral_spacing = self._signals.shape[1]

        self._m2mm_conversion_factor = 1000

        self._signals_interp_lat = self.lateral_interpolation()
        self._signals_interp = self.axial_interpolation()
        self._signals_grid_mm= self.px2mm_mesh()

    def lateral_interpolation(self):
        # CAUTION: to interpolate choose a smaller grid spacing NOT a larger grid!!
        grid_spacing = 1 / self._lateral_interpolation_factor
        lateral_spacing_interp_idx = np.arange(0,
                                               self._lateral_spacing,
                                               grid_spacing)  # new set of point indices in lateral (x) direction

        function_interp2d = RectBivariateSpline(self._axial_spacing_idx,
                                                self._lateral_spacing_idx,
                                                self._signals)

        signals_interp = function_interp2d(self._axial_spacing_idx, lateral_spacing_interp_idx)

        self._axial_spacing = signals_interp.shape[0]
        self._axial_spacing_idx = np.arange(0, self._axial_spacing, 1)
        self._lateral_spacing = signals_interp.shape[1]
        self._lateral_spacing_idx = np.arange(0, self._lateral_spacing, 1)

        return signals_interp
    
    def axial_interpolation(self):
        # CAUTION: to interpolate choose a smaller grid spacing NOT a larger grid!!
        grid_spacing = 1 / self._axial_interpolation_factor
        axial_spacing_interp_idx = np.arange(0,
                                               self._axial_spacing,
                                               grid_spacing)  # new set of point indices in lateral (x) direction

        function_interp2d = RectBivariateSpline(self._axial_spacing_idx,
                                                self._lateral_spacing_idx,
                                                self._signals_interp_lat)

        signals_interp = function_interp2d(axial_spacing_interp_idx, self._lateral_spacing_idx)

        self._axial_spacing = signals_interp.shape[0]
        self._axial_spacing_idx = np.arange(0, self._axial_spacing, 1)
        self._lateral_spacing = signals_interp.shape[1]
        self._lateral_spacing_idx = np.arange(0, self._lateral_spacing, 1)

        return signals_interp
    



    def px2mm_mesh(self):
        aperture_x_mm = (self._active_aperture_size * self._m2mm_conversion_factor)
        grid_x_conversion_px2mm = aperture_x_mm / self._lateral_spacing
        vector_x = np.arange(0, aperture_x_mm, grid_x_conversion_px2mm)

        aperture_z_mm = (self._recorded_depth * self._m2mm_conversion_factor)
        grid_z_conversion_px2mm = aperture_z_mm / self._axial_spacing
        vector_z = np.arange(0, aperture_z_mm, grid_z_conversion_px2mm)

        self.grid_conversion_px2mm = [grid_x_conversion_px2mm, grid_z_conversion_px2mm]

        return [vector_x, vector_z]

    @property
    def signals_lateral_interp(self):
        return self._signals_interp

    @property
    def imagegrid_mm(self):
        return self._signals_grid_mm

