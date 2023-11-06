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


# Delay table (Broadcast Implementation):
# shape: [1.imaging depth, 2.td_element (or lateral pixel coordinates), 3.td_element, 4.nbr of angles]
#
# 1st Dimension: Imaging depth
# 2nd Dimension: Distance/Time from one element to all other elements
# 3rd Dimension: 2. for all elements
# 4th Dimension: number of angles


import numpy as np
from datetime import datetime

class planewave_delays():
    def __init__(self, medium=None, sos=1540, fsampling=1, angles=0, max_time_sample = 1360):
        self._medium = medium
        self._speed_of_sound = sos
        self._sampling_frequency = fsampling
        self._angles = angles
        self._max_time_sample = max_time_sample - 1
        self._axial_pos_first_active_element()
        self._delay_table = self.delays_by_sample()

    def _axial_pos_first_active_element(self):
        axial_position = np.sign(self._angles) * np.max(self._medium[0])
        return axial_position.reshape(-1,1)

    def tx_dist2echo(self):
        # Distance calculation between the synthetic (delayed) PW transducer element and the point echo.
        # Accoustic wave travels in a single! "straight line" from the element to the point source
        #
        # broadcasted implementation:
        # [z, td_elements, angles] = [z, 1, angles] + [1, td_elements, angles]
        #
        # The geometrical problem is defined by a tilted (angle alpha) straight line (plane wave origin)
        # and the point echo source of the medium.
        #
        # dist_element2echo = z_echo * cos(alpha) + (x_0 - x_echo) * sin(alpha)
        #
        ###--->
        # dist_tx_element2echo = np.expand_dims(np.multiply(self._medium[1], np.cos(self._angles).reshape(1,-1)), axis=1) + \
        #                     np.expand_dims(np.multiply((self._axial_pos_first_active_element() - self._medium[0]), np.sin(self._angles)).T, axis=0)
        #
        # # Tx echos are the same for each echo point / td element combination
        # # Add a fourth dimension (td elements) and broadcast the array by tiling all data, finally adapt the shape.
        # # shape [depth, td_element (or lateral pixel coordinates), td_element, nbr of angles]
        # dist_tx_element2echo = np.expand_dims(dist_tx_element2echo, axis=3)
        # dist_tx_element2echo = np.tile(dist_tx_element2echo, self._medium[0].size)
        # dist_tx_element2echo = np.moveaxis(dist_tx_element2echo, 2, -1)
        ###--->
        echo_coords_axial = np.expand_dims(np.tile(self._medium[1], self._medium[0].size), axis=2)
        echo_coords_axial = np.repeat(echo_coords_axial, self._medium[0].size, axis=2)

        echo_coords_lateral = np.expand_dims(np.tile(self._medium[0].T, self._medium[1].size).T, axis=2)
        echo_coords_lateral = np.repeat(echo_coords_lateral, self._medium[0].size, axis=2)

        dist_tx_element2echo = echo_coords_axial * np.cos(self._angles) + echo_coords_lateral * np.sin(self._angles)

        # broadcast values to the total number of angles (rx delays are not affected by angle)
        # shape [depth, td_element (or lateral pixel coordinates), td_element, nbr of angles]
        dist_tx_element2echo = np.tile(np.expand_dims(dist_tx_element2echo, axis=3), self._angles.size)
        return dist_tx_element2echo.astype(np.float32)

    def rx_dist2echo(self):

        echo_coords_axial = np.expand_dims(np.tile(self._medium[1], self._medium[0].size), axis=2)
        echo_coords_axial = np.repeat(echo_coords_axial, self._medium[0].size, axis=2)

        echo_coords_lateral = np.expand_dims(np.tile(self._medium[0].T, self._medium[1].size).T, axis=2)

        transducer_element_coords_lateral = np.expand_dims(np.tile(self._medium[0].T, self._medium[0].size), axis=0)
        transducer_element_coords_lateral = np.repeat(transducer_element_coords_lateral, self._medium[1].size, axis=0)
        transducer_element_coords_lateral = np.moveaxis(transducer_element_coords_lateral,1,-1)

        dist_rx_echo2element = np.sqrt(echo_coords_axial ** 2 + (echo_coords_lateral - transducer_element_coords_lateral) ** 2)

        # broadcast values to the total number of angles (rx delays are not affected by angle)
        # shape [depth, td_element (or lateral pixel coordinates), td_element, nbr of angles]
        dist_rx_echo2element = np.tile(np.expand_dims(dist_rx_echo2element, axis=3), self._angles.size)

        return dist_rx_echo2element.astype(np.float32)


    def delays_by_sample(self):
        start_timing = datetime.now()

        delays_time = ((self.tx_dist2echo() + self.rx_dist2echo()) / self._speed_of_sound)
        delays_sample = np.rint(np.multiply(delays_time, self._sampling_frequency))
        delays_sample[delays_sample > self._max_time_sample] = 0

        end_timing = datetime.now()
        timing_delta_delaytable = end_timing - start_timing
        print(f'Time to initialize delay tables: {timing_delta_delaytable.total_seconds()} [s]')
        return delays_sample.astype(np.int32)


    @property
    def sample_delays(self):
        return self._delay_table