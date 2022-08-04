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

class tg_compensation():
    def __init__(self, signals=None, medium=None, center_frequency=None, cntrl_points=None, mode='points'):
        self._signals = signals
        self._center_frequency = center_frequency
        self._alpha = medium.alpha
        self._alpha_power = medium.alpha_power
        self._control_points = cntrl_points.tgc_control_points

        self._dB2neper = 8.686
        self._cm2m = 100

        if mode == 'points':
            self._tgc_signals = self.tgc_from_control_points()
        elif mode == 'alpha':
            self._tgc_signals = self.tgc_from_alpha()
        else:
            print(r'ERROR MESSAGE: Selected tgc type does not exist. Choose either \textit{points} or \textit{alpha}.')


    def tgc_from_control_points(self):
        # Function of the Control Point Function:
        #
        # Research Ultrasound systems provide digital control points (similar to the switches on a physical machine
        # to adjust a weight curve manually. This function weights the signals by interpolating the values between
        # the provided control points and multiplies them with the RF-signals of each channel.

        # Extrapolate TGC Control Points to total numer of recorded samples
        TGC_wave_idx = np.arange(0, self._signals.shape[0], 1)
        tgc_wave_idx = TGC_wave_idx[::self._signals.shape[0] // self._control_points.shape[1]]
        tgc_waveform = np.squeeze(self._control_points)
        TGC_waveform = np.interp(TGC_wave_idx, tgc_wave_idx, tgc_waveform)

        # bring into standardized shape for broadcasting
        TGC_waveform = np.repeat(np.expand_dims(TGC_waveform, axis=1), self._signals.shape[1], axis=1)
        TGC_waveform = np.repeat(np.expand_dims(TGC_waveform, axis=2), self._signals.shape[2], axis=2)

        return self._signals * TGC_waveform

    @property
    def signals(self):
        return self._tgc_signals