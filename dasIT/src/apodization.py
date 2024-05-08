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

class apodization():
    def __init__(self, delays=None, medium=None, transducer=None, apo='rec', angles=0):
        self._delays = delays
        self._medium = medium
        self._pwangles = transducer.planewaves_nr
        self._pitch = transducer.element_pitch
        self._nr_elements = transducer.transducer_elements
        self._pw_active_aperture = transducer.pw_aperture
        self._angles = angles


        if transducer.fnumber:
            self._fnumber = transducer.fnumber
        elif transducer.elevation_focus:
            self._fnumber = 1 / (2 * ((transducer.pw_aperture / 2) / transducer.elevation_focus))
        else:
            self._fnumber = 1.7

        self._apodization_type = apo
        if self._apodization_type == 'blackman':
            self._window = 'blackman'
        elif self._apodization_type == 'henning':
            self._window = 'henning'
        else:
            print(r'ERROR MESSAGE: Selected aodization type does not exist. Choose either \textit{rec} or \textit{blackman}.')
        self._apo_table = self.__apodization()


    def _round_elements(self, elements=None, type='odd'):
        if type == 'odd':
            return np.ceil(elements) // 2 * 2 + 1
        elif type == 'even':
            rounded_elements = np.ceil(elements) // 2 * 2
            # always start with at least 2 contributing elements
            rounded_elements = (rounded_elements + 2) if rounded_elements[0] == 0 else rounded_elements
            return rounded_elements
        else:
            print(r'ERROR MESSAGE: Wrong rounding argument in apodization.')

    def __apodization(self):
        # partially adapted from:
        # [1] C.L.Palmer and O.M.H.Rindal, Wireless, real - time plane - wave coherent compounding on an iphone - a
        # feasibility study, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 66, 7, pp.
        # 1222–1231, 2019. doi: https://doi.org/10.1109/TUFFC.2019.2914555
        # Calculate the array directivity based on the focus number of the transducer:


        px_grid_depth = self._medium[1]
        # active aperture = z / (2 * f)
        directive_aperture = (px_grid_depth / (2 * self._fnumber))
        # calculate how many elements are in this active aperture
        directive_aperture = (directive_aperture * self._medium[0].size) / self._pw_active_aperture
        # round to integer to get the number of active elements for each depth
        directive_aperture = self._round_elements(elements=directive_aperture, type='odd')
        # find the center of the active aperture
        directive_centre = int(np.amax(directive_aperture) / 2)

        # Create the apodization Kernel
        # Setup window function
        window = self.__create_window(sort=directive_aperture)

        # Initialize Kernel
        apo_kernel = np.zeros((directive_aperture.shape[0], int(np.amax(directive_aperture))))

        # Build Kernel
        for row_idx,(apo_datarow, directive_datarow, win) in enumerate(zip(apo_kernel, directive_aperture, window)):
            # Pad the active aperture (directivity) by setting the active element weights
            padding_start_element = int((len(apo_datarow) / 2) - (directive_datarow / 2))
            apo_kernel[row_idx, padding_start_element:(padding_start_element+len(win))] = win

        # Initialize the size of the apodization masks and pad to fit the kernel on the array boundaries
        # Initialize Mask
        apo_mask = np.zeros((self._medium[1].size, self._medium[0].size))
        # Pad Mask by filling up left and right sides of the array with the size of the directive center index
        # We need that to be able to slide a kernal from left to right
        # After the sliding additional areas not fitting the transducer spacing will be clipped off
        apo_mask = np.pad(apo_mask,
                          ((0, 0), (directive_centre , directive_centre)),
                          mode='constant',
                          constant_values=0)

        # Slide Kernel over apodization mask and clip the additional boarder area
        apo_mask = np.vstack([[np.insert(apo_mask, [0 + ch_idx], apo_kernel, axis=1)] for ch_idx in range(self._nr_elements)])
        apo_mask = apo_mask[: ,: , directive_centre:(self._nr_elements + directive_centre)]

        # Bring apodization mask into a standardized shape
        apo_mask = np.expand_dims(np.moveaxis(apo_mask, 0, -1), axis=3)

        return apo_mask.astype(np.int32)


    def __create_window(self, sort=None):
        print()
        if self._window == 'henning':
            window = [np.hanning(w) for w in np.squeeze(sort)]
        elif self._window == 'blackman':
            window = [np.blackman(w) for w in np.squeeze(sort)]
        else:
            print(r'ERROR MESSAGE: Wrong argument for window.')
        return window


    @property
    def table(self):
        return self._apo_table