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

class RXbeamformer():
    def __init__(self, signals=None, delays=None, apodization=None):
        self._signals = signals
        self._apodization = apodization
        self._delays = delays
        self._frame = self.beamform()

    def beamform(self):
        delays_depth_shape, delays_tdelement_px_shape, delays_tdelement_shape, delays_angles_shape = self._delays.shape
        _, tdelement_px_selector, tdelement_selector, angle_selector = np.ogrid[:0, :delays_tdelement_px_shape,:delays_tdelement_shape, :delays_angles_shape]

        # Delay, Apodize and Sum Signals
        # Delay tables select elements per channel
        frame = np.squeeze(np.sum(
            self._signals[self._delays, tdelement_px_selector, tdelement_selector, angle_selector] * self._apodization,
            axis=1))

        return frame


    @property
    def frame(self):
        return self._frame

