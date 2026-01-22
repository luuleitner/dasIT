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
from scipy.signal import firwin, convolve, get_window, welch, hilbert, resample


class RFfilter():
    def __init__(self, signals=None, fcutoff_band=None, fsampling=None, type='gaussian', order=None):
        self._signals = signals
        self._fcutoff_band = fcutoff_band
        self._fsampling = fsampling
        self._ftype = type
        self._forder = order

        self._signal_filtered = self.filter_signal()

    def filter_signal(self):
        signal = convolve(self._signals,
                          self.bandpass_firwin(),
                          mode='same')
        return signal

    def bandpass_firwin(self):
        # Define Gaussian Window
        std = 2.5 # MATLAB Standard
        win = get_window((self._ftype, std), self._forder)

        # Create Filter Coefficients
        filCoeff = firwin(self._forder,
                          [self._fcutoff_band[0], self._fcutoff_band[1]],
                          window=(self._ftype, win),
                          pass_zero=False,
                          scale=False,
                          fs=self._fsampling)
        filCoeff = np.broadcast_to(filCoeff[:, np.newaxis, np.newaxis], (self._forder, 1, 1)) # Broadcast Filter it in Original Data Shape
        return filCoeff

    @property
    def signal(self):
        return self._signal_filtered


def fftsignal(signal, f_sampling):
    f_spec, Pwr_den = welch(signal, f_sampling, nperseg=1024)
    return f_spec / 10**6, Pwr_den / 1000


def analytic_signal(signal, interp=False):
    hilbert_transformed_signal = hilbert(signal, axis=0)
    if interp:
        hilbert_transformed_signal_interp = resample(hilbert_transformed_signal, hilbert_transformed_signal.shape[0] * 3, axis=-1)
        return hilbert_transformed_signal_interp
    else:
        return hilbert_transformed_signal

def envelope(signal):
    return abs(signal)

def logcompression(signal, dbrange):
    # Adapted from:
    # [1] C. L. Palmer and O. M. H. Rindal, Wireless, real-time plane-wave coherent compounding on an iphone
    # - a feasibility study, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 66,
    # 7, pp. 1222–1231, 2019. doi: https://doi.org/10.1109/TUFFC.2019.2914555

    logcomp_signal = (20 * np.log10(envelope(signal))) - np.nanmax((20 * np.log10(envelope(signal))))
    logcomp_signal = np.where(logcomp_signal >= (-1 * dbrange), logcomp_signal, (-1 * dbrange))
    logcomp_signal = np.rint((255 * (logcomp_signal + dbrange)) / dbrange)

    return logcomp_signal.astype(np.float64)
